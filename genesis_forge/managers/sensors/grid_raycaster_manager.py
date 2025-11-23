from __future__ import annotations

import torch
import genesis as gs

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.sensors.base_sensor_manager import BaseSensorManager
from typing import TYPE_CHECKING,Sequence

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity

class GridRaycasterManager(BaseSensorManager):
    """
    A grid Lidar attached to an entity's link in the environment.

    Args:
        sensor_name: The name of the sensor
        env: The environment to sense.
        link_name: The name of the link the sensor is attached to
        entity_attr: The environment attribute which contains the entity with the links we're tracking. Defaults to `robot`.
        pos_offset: The xyz position offset between the link origin and the sensor origin
        euler_offset: The xyz orintation offset between the link origin and the sensor origin
        delay: The daly induced between sensor measurements
        draw_debug: Boolean value indicating whether the sensors data should be visualised in the scene
        read_frequency: the frequncy at which to read the input from the sensor in hz
        resolution : float
            Grid spacing in meters.
        size : tuple[float, float]
            Grid dimensions (length, width) in meters.
        direction : tuple[float, float, float]
            Ray direction vector.
        min_range : float, optional
            The minimum sensing range in meters. Defaults to 0.0.
        max_range : float, optional
            The maximum sensing range in meters. Defaults to 20.0.
        no_hit_value : float, optional
            The value to return for no hit. Defaults to max_range if not specified.
        return_world_frame : bool, optional
            Whether to return points in the world frame. Defaults to False (local frame).
        debug_sphere_radius: float, optional
            The radius of each debug sphere drawn in the scene. Defaults to 0.02.
        debug_ray_start_color: float, optional
            The color of each debug ray start sphere drawn in the scene. Defaults to (0.5, 0.5, 1.0, 1.0).
        debug_ray_hit_color: float, optional
            The color of each debug ray hit point sphere drawn in the scene. Defaults to (1.0, 0.5, 0.5, 1.0).
    """
    #TODO: write proper example

    def __init__(
        self,
        env: GenesisEnv,
        sensor_name: str,
        link_name: str,
        entity_attr: RigidEntity = "robot",
        pos_offset: list[float] = [0,0,0],
        euler_offset: list[float] = [0,0,0],
        delay: float = 0.0,
        draw_debug: bool = False,
        read_frequency: float= 25,
        resolution : float=0.1,
        size : tuple[float, float]=(1,1),
        direction : tuple[float, float, float]=(0,0,1),
        min_range: float = 0.0,
        max_range: float = 20.0,
        no_hit_value: float = None,
        return_world_frame: bool = False,
        debug_sphere_radius: float = 0.02,
        debug_ray_start_color: tuple[float, float, float, float] = (0.5, 0.5, 1.0, 1.0),
        debug_ray_hit_color: tuple[float, float, float, float] = (1.0, 0.5, 0.5, 1.0),
    ):
        super().__init__(            
            env=env,
            sensor_name=sensor_name,
            link_name=link_name,
            entity_attr=entity_attr,
            pos_offset=pos_offset,
            euler_offset=euler_offset,
            delay=delay,
            draw_debug=draw_debug,
            read_freuency=read_frequency)
        
        # scene=env.scene
        grid_pattern=gs.sensors.GridPattern(
            resolution=resolution,
            size=size,
            direction=direction,
        )
        self._spherical_raycaster_sensor = self.env.scene.add_sensor(
            gs.sensors.Lidar(
                pattern=grid_pattern,
                entity_idx=self._entity.idx,
                link_idx_local=self._link.idx_local,
                pos_offset=self._pos_offset,
                euler_offset=self._euler_offset,
                delay=self._delay,
                draw_debug=self._draw_debug,
                min_range=min_range,
                max_range=max_range,
                no_hit_value=no_hit_value,
                return_world_frame=return_world_frame,
                debug_sphere_radius=debug_sphere_radius,
                debug_ray_start_color=debug_ray_start_color,
                debug_ray_hit_color=debug_ray_hit_color
            )
        )

    """
    Properties
    """

    @property
    def link_name(self) -> torch.Tensor:
        """name of the imu link."""
        return self._link_name
    
    @property
    def link_idx(self) -> torch.Tensor:
        """The link index for the imu link."""
        return self._link.idx
    
    @property
    def local_link_idx(self) -> torch.Tensor:
        """The local link index for the imu link."""
        return self._link.idx_local

    """
    Helper Methods
    """
    def get_points(self) -> torch.Tensor:
        """
        Get the point cloud measured by the sensor 

        Returns:
            The point cloud shape is (n_envs,n_points, 3)
        """
        return self._sensor_reading[0]
    
    def get_diatnaces(self) -> torch.Tensor:
        """
        Get the distances measured by the sensor 

        Returns:
            The distances shape is (n_envs,n_points)
        """
        return self._sensor_reading[0]


    """
    Lifecycle Operations
    """

    def build(self):
        """Initialize link indices and buffers."""
        super().build()
        self._num_points=self.sensor.read().shape[1]
        self._sensor_reading=torch.zeros(self.env.num_envs,self._num_points,3)

    def reset(self, envs_idx: list[int] | None = None):
        super().reset(envs_idx)
        if not self.enabled:
            return
        
        self._sensor_reading[self.envs_idx,:,:]=0.0

    def step(self):
        super().step()
        if not self.enabled:
            return
        
        if self._last_reading_timestamp is None:
            self._last_reading_timestamp=self.env.scene.cur_t
        elif self.env.scene.cur_t-self._last_reading_timestamp>self._sensor_read_interval:
            self._sensor_reading[:]=self._spherical_raycaster_sensor.read()
            self._last_reading_timestamp=self.env.scene.cur_t
