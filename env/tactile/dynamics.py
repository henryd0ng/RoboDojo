"""Keep native scene objects and the tactile solver on one authoritative pose."""

from __future__ import annotations

import numpy as np


def restore_robot_state(robot_manager, simulation):
    """Restore actual joints and publish their poses before creating a world.

    Isaac's ordinary robot reset sets targets only. A new contact world must
    start at the restored state, rather than sweep from the last episode's
    grasp while the task objects have already returned to their initial poses.
    """
    for robot in robot_manager.robot_key:
        robot.reset()
        robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    simulation.scene.write_data_to_sim()
    simulation.sim.forward()
    simulation.scene.update(dt=0.0)
    # forward() updates Fabric but not USD. The PhysX update_transformations
    # API also leaves USD stale with this GPU/Fabric setup. Publish actual
    # link poses explicitly before TacEx reads attachment offsets from USD.
    from omni.usd import get_context

    stage = get_context().get_stage()
    for robot in robot_manager.robot_key:
        ArticulationRenderBridge(robot, stage).sync(fabric=False)


class ArticulationRenderBridge:
    """Publish every measured link pose to USD and its Fabric descendants.

    PhysX remains authoritative. This includes rigid links without collision
    geometry, such as an assembled wrist camera, rather than only IPC shells.
    """

    def __init__(self, robot, stage):
        from pxr import Usd, UsdGeom, UsdPhysics
        from transforms3d.quaternions import quat2mat
        from .mount import _single_env_path

        self.robot, self.stage = robot, stage
        root = stage.GetPrimAtPath(_single_env_path(robot.cfg.prim_path))
        if not root.IsValid():
            raise ValueError(f"Missing articulation USD root: {robot.cfg.prim_path}")
        bodies = {}
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                bodies.setdefault(prim.GetName(), []).append(prim)
        self._links = []
        cache = UsdGeom.XformCache()
        for index, name in enumerate(robot.body_names):
            matches = bodies.get(name, ())
            if len(matches) != 1:
                raise ValueError(f"{root.GetPath()}: measured body {name!r} has {len(matches)} USD matches")
            prim = matches[0]
            initial = np.asarray(cache.GetLocalToWorldTransform(prim)).T
            _, orientation = rigid_pose(initial)
            stretch = quat2mat(orientation).T @ initial[:3, :3]
            self._links.append((index, prim, stretch))
        self._links.sort(key=lambda item: item[1].GetPath().pathElementCount)

    def sync(self, link_transforms=None, *, fabric=True):
        from pxr import Gf, Usd, UsdGeom
        from transforms3d.quaternions import quat2mat

        if link_transforms is None:
            link_transforms = self.robot.root_physx_view.get_link_transforms().detach().cpu().numpy()
        poses = np.asarray(link_transforms)[0]
        if poses.shape != (len(self._links), 7) or not np.isfinite(poses).all():
            raise ValueError("Measured articulation poses must be finite (num_bodies, 7) xyz/xyzw values")
        cache, worlds = UsdGeom.XformCache(), {}
        for index, prim, stretch in self._links:
            live = poses[index]
            world = np.eye(4)
            world[:3, :3] = quat2mat([live[6], *live[3:6]]) @ stretch
            world[:3, 3] = live[:3]
            xform = UsdGeom.Xformable(prim)
            reset_stack = xform.GetResetXformStack()
            parent = np.eye(4) if reset_stack else np.asarray(cache.GetLocalToWorldTransform(prim.GetParent())).T
            attribute = prim.GetAttribute("xformOp:transform:tactileReset")
            xform.ClearXformOpOrder()
            op = UsdGeom.XformOp(attribute) if attribute else xform.AddTransformOp(opSuffix="tactileReset")
            xform.SetXformOpOrder([op], resetXformStack=reset_stack)
            op.Set(Gf.Matrix4d(np.linalg.solve(parent, world).T.tolist()))
            cache.Clear()
            actual = np.asarray(cache.GetLocalToWorldTransform(prim)).T
            if not np.allclose(actual, world, atol=1e-7, rtol=1e-7):
                raise RuntimeError(f"Articulation USD/PhysX transform mismatch: {prim.GetPath()}")
            worlds[str(prim.GetPath())] = world
        if fabric:
            _sync_fabric_link_transforms(worlds)


def _sync_fabric_link_transforms(worlds):
    """Use Fabric's hierarchy API so cameras and meshes follow each link."""
    from omni.usd import get_context
    import usdrt

    stage = usdrt.Usd.Stage.Attach(get_context().get_stage_id())
    hierarchy = usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(
        stage.GetFabricId(), stage.GetStageIdAsStageId()
    )
    for path, world in worlds.items():
        matrix = usdrt.Gf.Matrix4d(*world.T.ravel().tolist())
        if hierarchy.set_world_xform(usdrt.Sdf.Path(path), matrix) is False:
            raise RuntimeError(f"Failed to publish articulation pose to Fabric: {path}")
    hierarchy.update_world_xforms()


def rigid_pose(matrix):
    """Extract a proper rotation from an affine body without leaking its scale."""
    from transforms3d.quaternions import mat2quat

    matrix = np.asarray(matrix, dtype=float).reshape(4, 4)
    if not np.isfinite(matrix).all():
        raise RuntimeError("The tactile solver produced a non-finite object transform.")
    left, _, right = np.linalg.svd(matrix[:3, :3])
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(left @ right)
    rotation = left @ correction @ right
    return matrix[:3, 3].copy(), mat2quat(rotation)


def create_driven_body(config, simulation, *, strength=100.0):
    """Create a robot contact shell with continuous constrained motion.

    A transform constraint lets IPC resolve motion over the step. Repeated
    teleportation of a fixed collider would bypass that collision trajectory.
    """
    from tacex_uipc import UipcRigidObject
    from uipc import Vector2
    from uipc.constitution import SoftTransformConstraint

    class DrivenBody(UipcRigidObject):
        def _setup_uipc_mesh(self):
            mesh = super()._setup_uipc_mesh()
            SoftTransformConstraint().apply_to(mesh, Vector2.Values([strength, strength]))
            return mesh

    return DrivenBody(config, simulation)


class UipcRobotBody:
    """Drive an IPC contact shell from the actual PhysX articulation pose.

    Reaction forces remain in IPC; they are not fed back to the robot actuator.
    Tracking error is exposed so this approximation can be checked explicitly.
    """

    def __init__(self, binding, body, simulation):
        self.binding = binding
        self.body = body
        self.robot = binding.robot_key
        indices, _ = self.robot.find_bodies(binding.body_name, preserve_order=True)
        if len(indices) != 1:
            raise ValueError(f"Expected one robot body for {binding.body_name!r}, got {indices}")
        self.body_index = indices[0]
        self.env_index = binding.scene_object.env_id
        self.target = np.asarray(body.geo_slot_list[0].geometry().transforms().view()).reshape(-1, 4, 4)[0].copy()
        simulation.scene.animator().insert(body.uipc_scene_objects[0], self._animate)

    def set_target(self, link_transforms):
        from transforms3d.quaternions import quat2mat

        pose = link_transforms[self.env_index, self.body_index]
        self.target[:3, 3] = pose[:3]
        self.target[:3, :3] = quat2mat([pose[6], *pose[3:6]])

    def _animate(self, info):
        from uipc import builtin, view

        geometry = info.geo_slots()[0].geometry()
        view(geometry.instances().find(builtin.is_constrained))[0] = 1
        view(geometry.instances().find(builtin.aim_transform))[0] = self.target

    def tracking_error_m(self):
        current = np.asarray(self.body.geo_slot_list[0].geometry().transforms().view()).reshape(-1, 4, 4)[0]
        return float(np.linalg.norm(current[:3, 3] - self.target[:3, 3]))


def _native_actor_view(root_path, stage):
    from isaacsim.core.simulation_manager import SimulationManager
    from pxr import Usd, UsdPhysics

    simulation = SimulationManager.get_physics_sim_view()
    if simulation is None or not simulation.is_valid:
        raise RuntimeError("Native PhysX view must be valid before transferring object ownership")
    # Wildcards also match visual/material prims and produce PhysX errors for
    # singleton bodies. USD only narrows candidates; the returned tensor view
    # remains authoritative about which rigid actors actually exist.
    candidates = [root_path]
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if (str(prim.GetPath()) != root_path and prim.HasAPI(UsdPhysics.RigidBodyAPI)
                and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not False):
            candidates.append(str(prim.GetPath()))
    return simulation, simulation.create_rigid_body_view(candidates)


def _attribute_opinion(attribute):
    spec = attribute.GetStage().GetEditTarget().GetPropertySpecForScenePath(attribute.GetPath())
    return attribute, bool(spec), bool(spec and spec.HasDefaultValue()), spec.default if spec else None


def _restore_attribute_opinion(saved):
    attribute, existed, authored, value = saved
    if not existed:
        attribute.GetPrim().RemoveProperty(attribute.GetName())
    elif authored:
        attribute.Set(value)
    else:
        attribute.Clear()


class UipcSceneBody:
    """Expose UIPC motion through the existing RoboDojo rigid-object wrapper.

    The original actor remains as a collision-free kinematic visual follower.
    Removing a live PhysX actor invalidates the shared articulation view, so
    UIPC owns its motion while the existing actor only mirrors that pose.
    """

    def __init__(self, binding, body, world, device):
        from pxr import Usd, UsdGeom, UsdPhysics
        from transforms3d.quaternions import quat2mat

        self.binding = binding
        self.body = body
        self.world = world
        self.device = device
        self.wrapper = binding.scene_object.wrapper
        self.stage = self.wrapper.stage
        self.prim = self.stage.GetPrimAtPath(binding.scene_object.prim_path)
        self._physics_attributes = []
        self._actor_xforms = []
        self._closed = False
        self._xform = UsdGeom.Xformable(self.prim)
        self._reset_xform_stack = self._xform.GetResetXformStack()
        self._native_simulation, self._native_view = _native_actor_view(str(self.prim.GetPath()), self.stage)
        self.native_actor_paths = tuple(self._native_view.prim_paths)
        root_path = str(self.prim.GetPath())
        if (root_path not in self.native_actor_paths or len(set(self.native_actor_paths)) != len(self.native_actor_paths)
                or any(path != root_path and not path.startswith(root_path + "/") for path in self.native_actor_paths)):
            raise RuntimeError(f"Unexpected native actor set for {root_path}: {self.native_actor_paths}")
        native = self._native_view.get_transforms()
        self._tensor_template = native
        self._initial_native_transforms = self._numpy(native).copy()
        self._initial_native_velocities = self._numpy(self._native_view.get_velocities()).copy()
        self._initial_native_kinematic = np.asarray([
            bool(UsdPhysics.RigidBodyAPI(self.stage.GetPrimAtPath(path)).GetKinematicEnabledAttr().Get())
            for path in self.native_actor_paths
        ], dtype=bool)
        if self._initial_native_transforms.shape != (len(self.native_actor_paths), 7):
            raise RuntimeError("Native actor transform count does not match prim_paths")
        # Offsets belong to the measured native-preparation frame; a bounded
        # UIPC initialization correction must not be cancelled by child offsets.
        initial_root = np.eye(4)
        initial_root[:3, :3] = quat2mat(binding.initial_pose[3:])
        initial_root[:3, 3] = binding.initial_pose[:3]
        self._actor_offsets = []
        cache = UsdGeom.XformCache()
        for index, path in enumerate(self.native_actor_paths):
            native_pose = self._initial_native_transforms[index]
            native_world = np.eye(4)
            native_world[:3, :3] = quat2mat([native_pose[6], *native_pose[3:6]])
            native_world[:3, 3] = native_pose[:3]
            relative = np.linalg.solve(initial_root, native_world)
            if path == root_path:
                position_error = np.linalg.norm(relative[:3, 3])
                rotation_error = np.linalg.norm(relative[:3, :3] - np.eye(3))
                if position_error > 1e-4 or rotation_error > 1e-3:
                    raise RuntimeError(
                        f"Native/UIPC root pose mismatch before ownership transfer: {root_path} "
                        f"(position={position_error:.6g} m, rotation_matrix={rotation_error:.6g})"
                    )
                relative = np.eye(4)
            self._actor_offsets.append(relative)
            actor_prim = self.stage.GetPrimAtPath(path)
            xform = UsdGeom.Xformable(actor_prim)
            original_world = np.asarray(cache.GetLocalToWorldTransform(actor_prim)).T
            _, rotation = rigid_pose(original_world)
            suffix = "tactile" if path == root_path else "tactileFollower"
            if actor_prim.HasProperty(f"xformOp:transform:{suffix}"):
                raise RuntimeError(f"Stale tactile transform on native actor: {path}")
            self._actor_xforms.append({"index": index, "xform": xform, "prim": actor_prim,
                                      "reset": xform.GetResetXformStack(), "order": _attribute_opinion(xform.GetXformOpOrderAttr()),
                                      "stretch": quat2mat(rotation).T @ original_world[:3, :3], "suffix": suffix})

        try:
            # Switching a native dynamic actor to kinematic clears its velocity.
            # PhysX rejects velocity setters once that flag is enabled.
            for prim in Usd.PrimRange(self.prim):
                for schema, getter, value in (
                    (UsdPhysics.RigidBodyAPI, "GetKinematicEnabledAttr", True),
                    (UsdPhysics.CollisionAPI, "GetCollisionEnabledAttr", False),
                ):
                    if prim.HasAPI(schema):
                        attribute = getattr(schema(prim), getter)()
                        self._physics_attributes.append(_attribute_opinion(attribute))
                        attribute.Set(value)
            for actor in self._actor_xforms:
                xform = actor["xform"]
                xform.ClearXformOpOrder()
                xform.SetResetXformStack(actor["reset"])
                actor["op"] = xform.AddTransformOp(opSuffix=actor["suffix"])
            self.wrapper.set_dynamics_backend(self)
            self.sync_visual()
        except Exception:
            self.close()
            raise

    @staticmethod
    def _numpy(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value.numpy() if hasattr(value, "numpy") else value)

    def _native_tensor(self, value, *, indices=False):
        # PhysX may be on CPU while the contact solver is on CUDA. Match the
        # actual PhysX frontend and device, never the UIPC device argument.
        if hasattr(self._tensor_template, "detach"):
            import torch

            return torch.as_tensor(value, dtype=torch.int32 if indices else torch.float32,
                                   device=self._tensor_template.device).contiguous()
        if isinstance(self._tensor_template, np.ndarray):
            return np.ascontiguousarray(value, dtype=np.uint32 if indices else np.float32)
        import warp as wp

        return wp.array(value, dtype=wp.int32 if indices else wp.float32, device=self._tensor_template.device)

    def world_matrix(self):
        return np.asarray(self.body.geo_slot_list[0].geometry().transforms().view()).reshape(-1, 4, 4)[0].copy()

    def _parent_matrix(self):
        from pxr import Usd, UsdGeom

        if self._reset_xform_stack:
            return np.eye(4)
        return np.asarray(
            UsdGeom.Xformable(self.prim.GetParent()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        ).T

    def get_local_pose(self):
        return rigid_pose(np.linalg.solve(self._parent_matrix(), self.world_matrix()))

    def get_world_pose(self):
        return rigid_pose(self.world_matrix())


    def set_local_pose(self, translation=None, orientation=None):
        import torch
        from transforms3d.quaternions import quat2mat

        position, quaternion = self.get_local_pose()
        local = np.eye(4)
        local[:3, 3] = position if translation is None else np.asarray(translation)
        local[:3, :3] = quat2mat(quaternion if orientation is None else np.asarray(orientation))
        position, quaternion = rigid_pose(self._parent_matrix() @ local)
        pose = torch.tensor([*position, *quaternion], dtype=torch.float32, device=self.device).reshape(1, 7)
        self.body.write_pose_to_sim(pose, zero_velocity=True)
        self.world.retrieve()
        self.sync_visual()

    def clear_velocity(self):
        import torch

        self.body.write_velocity_to_sim(torch.zeros((1, 4, 4), dtype=torch.float32, device=self.device))

    def sync_visual(self):
        from pxr import Gf, UsdGeom
        from transforms3d.quaternions import quat2mat

        if not self._native_simulation.is_valid:
            raise RuntimeError("Native PhysX view became invalid while following UIPC")
        position, quaternion = rigid_pose(self.world_matrix())
        root = np.eye(4)
        root[:3, :3], root[:3, 3] = quat2mat(quaternion), position
        actor_worlds = [root @ offset for offset in self._actor_offsets]
        poses = []
        for transform in actor_worlds:
            actor_position, actor_quaternion = rigid_pose(transform)
            poses.append([*actor_position, *actor_quaternion[1:], actor_quaternion[0]])
        cache = UsdGeom.XformCache()
        for actor in sorted(self._actor_xforms, key=lambda record: record["prim"].GetPath().pathElementCount):
            transform = actor_worlds[actor["index"]].copy()
            transform[:3, :3] = transform[:3, :3] @ actor["stretch"]
            parent = (np.eye(4) if actor["reset"] else
                      np.asarray(cache.GetLocalToWorldTransform(actor["prim"].GetParent())).T)
            actor["op"].Set(Gf.Matrix4d(np.linalg.solve(parent, transform).T.tolist()))
            cache.Clear()
        # Fabric renders live PhysX poses even when USD has been updated.
        # This actor has no contacts or dynamics; never read its pose back
        # into the contact solver or native task observations.
        self._native_view.set_transforms(self._native_tensor(poses),
                                        self._native_tensor(np.arange(len(poses)), indices=True))

    def close(self):
        if self._closed:
            return
        self.wrapper.set_dynamics_backend(None)
        if self.prim.IsValid():
            for actor in self._actor_xforms:
                _restore_attribute_opinion(actor["order"])
                actor["prim"].RemoveProperty(f"xformOp:transform:{actor['suffix']}")
            # Re-enable SDF mesh shapes while actors are still kinematic;
            # PhysX rejects triangle-shape enabling after they become dynamic.
            for opinion in sorted(self._physics_attributes,
                                  key=lambda saved: saved[0].GetName() == "physics:kinematicEnabled"):
                _restore_attribute_opinion(opinion)
            if self._native_simulation.is_valid:
                indices = self._native_tensor(np.arange(len(self.native_actor_paths)), indices=True)
                self._native_view.set_transforms(self._native_tensor(self._initial_native_transforms), indices)
                dynamic_indices = np.flatnonzero(~self._initial_native_kinematic)
                if len(dynamic_indices):
                    # The tensor API uses a full-view buffer and selected indices.
                    self._native_view.set_velocities(self._native_tensor(self._initial_native_velocities),
                                                    self._native_tensor(dynamic_indices, indices=True))
        self._closed = True
