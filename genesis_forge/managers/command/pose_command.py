from typing import Tuple, TypedDict

import os
import torch
import genesis as gs
from genesis.utils.geom import euler_to_R
import numpy as np
from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.gamepads import Gamepad

from .command_manager import CommandManager, CommandRangeValue


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class PoseCommandRange(TypedDict):
    pos_x: CommandRangeValue
    pos_y: CommandRangeValue
    pos_z: CommandRangeValue


class PoseDebugVisualizerConfig(TypedDict):
    """Defines the configuration for the debug visualizer."""

    envs_idx: list[int]
    """The indices of the environments to visualize. If None, all environments will be visualized."""

    arrow_offset: float
    """The vertical offset of the debug arrows from the top of the robot"""

    arrow_radius: float
    """The radius of the shaft of the debug arrows"""

    commanded_color: Tuple[float, float, float, float]
    """The color of the commanded velocity arrow"""



DEFAULT_VISUALIZER_CONFIG: PoseDebugVisualizerConfig = {
    "envs_idx": None,
    "sphere_offset": 0.03,
    "sphere_radius": 0.02,
    "commanded_color": (0.0, 0.5, 0.0, 1.0),
}


class PoseCommandManager(CommandManager):
    """
    Generates a position command from uniform distribution.
    The command comprises of a linear velocity in x and y direction and an angular velocity around the z-axis.

    IMPORTANT: The position commands are interpreted as world-relative coordinates:
    - X-axis: x coordinate of the target position
    - Y-axis: y coordinate of the target position
    - Z-axis: z coordinate of the target position

    :::{admonition} Debug Visualization

        If you set `debug_visualizer` to True, target sphere will be rendered above the target pos

        Arrow meanings:

        - GREEN: Commanded position for the robot in the world frame

    Args:
        env: The environment to control
        range: The ranges of linear & angular velocities
        resample_time_sec: The time interval between changing the command
        debug_visualizer: Enable the debug arrow visualization
        debug_visualizer_cfg: The configuration for the debug visualizer

    Example::

        class MyEnv(GenesisEnv):
            def config(self):
                # Create a velocity command manager
                self.command_manager = PoseCommandManager(
                    self,
                    visualize=True,
                    range = {
                        "lin_vel_x_range": (-1.0, 1.0),
                        "lin_vel_y_range": (-1.0, 1.0),
                        "ang_vel_z_range": (-0.5, 0.5),
                    }
                )

                RewardManager(
                    self,
                    logging_enabled=True,
                    cfg={
                        "tracking_lin_vel": {
                            "weight": 1.0,
                            "fn": rewards.command_tracking_lin_vel,
                            "params": {
                                "vel_cmd_manager": self.velocity_command,
                            },
                        },
                        "tracking_ang_vel": {
                            "weight": 1.0,
                            "fn": rewards.command_tracking_ang_vel,
                            "params": {
                                "vel_cmd_manager": self.velocity_command,
                            },
                        },
                        # ... other rewards ...
                    },
                )

                # Observations
                ObservationManager(
                    self,
                    cfg={
                        "velocity_cmd": {"fn": self.velocity_command.observation},
                        # ... other observations ...
                    },
                )
    """

    def __init__(
        self,
        env: GenesisEnv,
        range: PoseCommandRange,
        resample_time_sec: float = 5.0,
        debug_visualizer: bool = False,
        debug_visualizer_cfg: PoseDebugVisualizerConfig = DEFAULT_VISUALIZER_CONFIG,
    ):
        super().__init__(env, range=range, resample_time_sec=resample_time_sec)
        self._sphere_nodes: list = []
        self.debug_visualizer = debug_visualizer
        self.visualizer_cfg = {**DEFAULT_VISUALIZER_CONFIG, **debug_visualizer_cfg}
        self.debug_envs_idx = None

        self._is_standing_env = torch.zeros(
            env.num_envs, dtype=torch.bool, device=gs.device
        )

    """
    Lifecycle Operations
    """

    def resample_command(self, env_ids: list[int]):
        """
        Overwrites commands for environments that should be standing still.
        """
        super().resample_command(env_ids)
        if not self.enabled:
            return

    def build(self):
        """Build the position command manager"""
        super().build()

        # If debug envs_idx is not set, attempt to use the vis_options rendered_envs_idx
        if not self.debug_visualizer or self.visualizer_cfg is None or self.env.scene is None:
            return        
        self.debug_envs_idx = self.visualizer_cfg.get("envs_idx", None)
        if self.debug_envs_idx is None and self.env.scene.vis_options is not None:
            self.debug_envs_idx = self.env.scene.vis_options.rendered_envs_idx
        if self.debug_envs_idx is None:
            self.debug_envs_idx = list[int](range(self.env.num_envs))

    def step(self):
        """Render the command arrows"""
        if not self.enabled:
            return
        super().step()
        self._render_arrow()
        
    def use_gamepad(
        self,
        gamepad: Gamepad,
        pos_x_axis: int = 0,
        pos_y_axis: int = 1,
        pos_z_axis: int = 2,
        euler_x_axis: int = 3,
        euler_y_axis: int = 4,
        euler_z_axis: int = 5,
    ):
        """
        Use a connected gamepad to control the command.

        Args:
            gamepad: The gamepad to use.
            pos_x_axis: Map this gamepad axis index to the position in the x-direction.
            pos_y_axis: Map this gamepad axis index to the position in the y-direction.
            pos_z_axis: Map this gamepad axis index to the position in the z-direction.
            euler_x_axis: Map this gamepad axis index to the euler in the x-direction.
            euler_y_axis: Map this gamepad axis index to the euler in the y-direction.
            euler_z_axis: Map this gamepad axis index to the euler in the z-direction.
        """
        super().use_gamepad(
            gamepad,
            range_axis={
                "pos_x": pos_x_axis,
                "pos_y": pos_y_axis,
                "pos_z": pos_z_axis,
                "euler_x": euler_x_axis,
                "euler_y": euler_y_axis,
                "euler_z": euler_z_axis,
            },
        )

    """
    Internal Implementation
    """

    def _render_arrow(self):
        """
        Render the command sphere showing position commands.

        The commanded position sphere (green) shows the position in the world frame 
        """
        if not self.debug_visualizer:
            return

        # Remove existing arrows
        for arrow in self._arrow_nodes:
            self.env.scene.clear_debug_object(arrow)
        self._arrow_nodes = []

        for i in self.debug_envs_idx:
            # Target arrow (robot-relative command transformed to world coordinates for visualization)
            self._draw_arrow(
                pos=self.command[i],
                color=self.visualizer_cfg["commanded_color"],
            )

    def _draw_arrow(
        self,
        pos: torch.Tensor,
        euler: torch.Tensor,
        color: list[float],
    ):
        try:
            node = self.env.scene.draw_debug_arrow(
                pos=pos.cpu().numpy(),
                vec=np.tile([0,0,1], (pos.shape[0], 1))@euler_to_R(euler),
                color=color,
                radius=self.visualizer_cfg["arrow_radius"],
            )
            if node:
                self._sphere_nodes.append(node)
        except Exception as e:
            print(f"Error adding debug visualizing in PoseCommandManager: {e}")
