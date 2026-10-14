import time

from isaaclab.envs.direct_rl_env import DirectRLEnv


class CustomDirectRLEnv(DirectRLEnv):
    def sim_step(self, render: bool = True):
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        tactile_manager = getattr(self, "tactile_manager", None)
        tactile_active = (
            tactile_manager is not None
            and tactile_manager.enabled
            and tactile_manager.initialized
            and not tactile_manager.suspended
        )
        if tactile_active:
            tactile_manager.ensure_initialized()

        for _ in range(self.cfg.decimation):
            started = time.perf_counter() if tactile_active else 0.0
            self._sim_step_counter += 1
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if tactile_active:
                # TacEx attachment callbacks read the current articulation
                # pose, so update IsaacLab buffers before advancing the gelpad
                # and publishing its render mesh.
                self.scene.update(dt=self.physics_dt)
                tactile_manager.physics_step(self.physics_dt, physx_elapsed_s=time.perf_counter() - started)
            if (
                self.cfg.sim.render_interval > 0
                and self._sim_step_counter % self.cfg.sim.render_interval == 0
                and is_rendering
                and render
            ):
                if tactile_active:
                    tactile_manager.before_render()
                self.sim.render()
            if not tactile_active:
                # Pending, disabled and suspended worlds retain native order.
                self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        return self.extras
