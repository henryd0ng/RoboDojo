from typing import Any, List
from uuid import uuid4

from env.camera_manager.camera_manager import CameraManager
from env.environment.base_env import BaseEnv
from env.robot_manager.control_manager import ControlSeq
from env.robot_manager.robot_manager import RobotManager
from env.scene_manager.scene_manager import SceneManager
from env.tactile import RoboDojoTactileManager


class TaskEnv(BaseEnv):
    def __init__(self, config, app, **kwargs):
        self.sim_config = config.sim
        self.scene_config = config.scene
        self.num_envs = self.sim_config.scene.num_envs
        self.env_spacing = self.sim_config.scene.env_spacing
        self.device = self.sim_config.get("device", "cpu")
        self.dt = self.sim_config.dt
        self._episode_namespace = uuid4().hex
        self._episode_number = 0
        self.episode_ids = []
        # This constructor is import-light. TacEx/UIPC are loaded only when
        # the explicit tactile.enabled flag is true during scene setup.
        self.tactile_manager = RoboDojoTactileManager(config.get("tactile", {}))

        # RobotManager must be created before BaseEnv.__init__ because it
        # provides the InteractiveSceneCfg used to launch the simulation.
        self.robot_config = config.robot
        self.robot_manager = RobotManager(
            num_envs=self.num_envs,
            env_spacing=self.env_spacing,
            config=self.robot_config,
            dt=self.dt,
            device=self.device,
        )
        self._interactiveSceneCfg = self.robot_manager._get_SceneCfg(
            num_envs=self.num_envs, env_spacing=self.env_spacing
        )
        super().__init__(self.sim_config, app=app, interactiveSceneCfg=self._interactiveSceneCfg, **kwargs)
        self.env_seed_list = self.env_seeds

        self.scene_manager = SceneManager(
            self.num_envs,
            self.device,
            self.sim_config.scene.env_spacing,
            use_fabric=self.use_fabric,
            seeds_per_env=self.env_seed_list,
            scene_config=self.scene_config,
            task_config=config.task_env,
        )

        self.robot_manager._load_camera_cfg(config.camera)
        self.capture_config = config.camera
        self.camera_config = config.camera
        self.camera_manager = CameraManager(
            self.num_envs,
            self.camera_config,
            self.device,
            seeds_per_env=self.env_seed_list,
        )
        from env.camera_manager.capture.tiled_capture_manager import TiledCaptureManager

        self.capture_manager = TiledCaptureManager(
            self.num_envs,
            self.capture_config,
            self.camera_manager,
            self.device,
        )

    def _setup_scene(self, sim):
        self.robot_manager.initialize(sim)
        super()._setup_scene(sim)

    def _post_setup_scene(self, sim):
        self.scene_manager.initialize(sim)
        super()._post_setup_scene(sim)
        self.camera_manager.initialize(sim)
        self.capture_manager.initialize(sim)
        self.camera_manager.post_init()
        if self.tactile_manager.enabled:
            # Rigid sensor housings belong to the articulation before PLAY.
            # The contact world needs the complete layout, built during reset.
            self.tactile_manager.mount(sim, self.robot_manager)
            sim.tactile_manager = self.tactile_manager

    def update_seed(self, seed: Any | None = None):
        super().update_seed(seed)
        if hasattr(self, "scene_manager") and self.scene_manager is not None:
            self.scene_manager.update_env_seeds(self.env_seeds)
        if hasattr(self, "robot_manager") and self.robot_manager is not None:
            self.robot_manager.update_env_seeds(self.env_seeds)
        if hasattr(self, "camera_manager") and self.camera_manager is not None:
            self.camera_manager.update_env_seeds(self.env_seeds)

    def reset(self, seed: Any | None = None, options: dict[str, Any] | None = None):
        self.tactile_manager.begin_reset()
        super().reset(seed=seed, options=options)
        self.scene_manager.reload_scene()
        self.robot_manager.reset()
        if self.tactile_manager.enabled:
            from env.tactile.dynamics import restore_robot_state

            restore_robot_state(self.robot_manager, self.sim)
            self._prepare_tactile_reset()
        if not self.tactile_manager.enabled or self.tactile_manager.initialized:
            self.tactile_manager.resume()
        self._episode_number += 1
        self.episode_ids = [
            f"{self._episode_namespace}:{self._episode_number}:{env_id}" for env_id in range(self.num_envs)
        ]
        settle_steps = 300
        if self.tactile_manager.enabled:
            if self.tactile_manager.initialized:
                settle_steps = max(settle_steps, self.tactile_manager.config.stabilise_steps)
            print(f"[RoboDojo tactile] settling {settle_steps} physics steps", flush=True)
        for index in range(settle_steps):
            self.sim_step(render=False)
            if self.tactile_manager.enabled and (index + 1) % 100 == 0:
                print(f"[RoboDojo tactile] settled {index + 1}/{settle_steps}", flush=True)
        # The first observation after reset must see the settled UIPC surface;
        # capture itself is deferred until ObsManager assembles that frame.
        self.tactile_manager.before_render()
        self.camera_manager.reset()
        self.capture_manager.reset()
        self._calibrate_tactile_reference()

    def _prepare_tactile_reset(self):
        """Prepare and transfer the native scene before standalone settling.

        Evaluation overrides this hook to retain its full native reset first.
        Robot state restoration remains in reset(), before either path steps.
        """
        self.scene_manager.apply_saved_poses(list(range(self.num_envs)))
        self._initialize_tactile_from_native_scene()

    def _initialize_tactile_from_native_scene(self):
        """Publish an already prepared native scene without integrating it."""
        from omni.physx import acquire_physx_interface

        acquire_physx_interface().update_transformations(False, True, True, False)
        self.tactile_manager.initialize(self.sim, self.robot_manager, self.scene_manager)

    def _calibrate_tactile_reference(self):
        if self.tactile_manager.enabled and self.tactile_manager.initialized:
            from env.camera_manager.capture.render_sync import wait_for_latest_cameras

            self.tactile_manager.calibrate_reference(lambda: wait_for_latest_cameras(self))

    def step(self, meta_control_list: List[ControlSeq]):
        self.robot_manager.control_robot(meta_control_list=meta_control_list)

    def close(self):
        self.tactile_manager.close()
        self.capture_manager.destroy()
        self.camera_manager.destroy()
        self.robot_manager.close()
        self.scene_manager.close()
        super().close()
