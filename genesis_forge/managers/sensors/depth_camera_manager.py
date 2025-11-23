from __future__ import annotations

import torch
import genesis as gs

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.sensors.base_sensor_manager import BaseSensorManager
from typing import TYPE_CHECKING,Sequence

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity

class DepthCameraManager(BaseSensorManager):
    """
    A Spherical Lidar attached to an entity's link in the environment.

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
        res: tuple[int, int]
            The resolution of the camera, specified as a tuple (width, height).
        fx : float | None
            Focal length in x direction in pixels. Computed from fov_horizontal if None.
        fy : float | None
            Focal length in y direction in pixels. Computed from fov_vertical if None.
        cx : float | None
            Principal point x coordinate in pixels. Defaults to image center if None.
        cy : float | None
            Principal point y coordinate in pixels. Defaults to image center if None.
        fov_horizontal : float
            Horizontal field of view in degrees. Used to compute fx if fx is None.
        fov_vertical : float | None
        Vertical field of view in degrees. Used to compute fy if fy is None.
        angles: tuple[Sequence[float], Sequence[float]], optional
            Array of horizontal/vertical angles. Overrides the other options if provided.
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
        res: tuple[int, int]=None,
        fx : float =None,
        fy : float =None,
        cx : float =None,
        cy : float =None,
        fov_horizontal : float=None,
        fov_vertical : float = None,
        min_range: float = 0.0,
        max_range: float = 20.0,
        no_hit_value: float = None,
        return_world_frame: bool = False,
        debug_sphere_radius: float = 0.02,
        debug_ray_start_color: tuple[float, float, float, float] = (0.5, 0.5, 1.0, 1.0),
        debug_ray_hit_color: tuple[float, float, float, float] = (1.0, 0.5, 0.5, 1.0),
        read_rays=False,
        read_images=False,
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
        depth_camera_pattern=gs.sensors.DepthCameraPattern(
            res=res,
            fx=fx,fy=fy,
            cx=cx,cy=cy,
            fov_horizontal=fov_horizontal,
            fov_vertical=fov_vertical
        )
        self._spherical_raycaster_sensor = self.env.scene.add_sensor(
            gs.sensors.Lidar(
                pattern=depth_camera_pattern,
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
        self._read_rays=read_rays
        self._read_images=read_images

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
        if self._read_rays:
            return self._sensor_cloud[0]
        gs.logger.error("the sensor was not initialised with the 'read_rays' arg, unable to get the points or distances")
        return None
    
    def get_diatnaces(self) -> torch.Tensor:
        """
        Get the distances measured by the sensor 

        Returns:
            The distances shape is (n_envs,n_points)
        """
        if self._read_rays:
            return self._sensor_cloud[1]
        gs.logger.error("the sensor was not initialised with the 'read_rays' option, unable to get the points or distances")
        return None
    
    def get_image(self)-> torch.Tensor:
        """
        Get the depth image from the sensor 

        Returns:
            The distances shape is (n_envs,n_points)
        """
        if self._read_images:
            return self._sensor_image
        gs.logger.error("the sensor was not initialised with the 'read_images' option, unable to get the depth image")
        return None


    """
    Lifecycle Operations
    """

    def build(self):
        """Initialize link indices and buffers."""
        super().build()
        self._num_points=self.sensor.read().shape[1]
        if self._read_rays:
            self._sensor_cloud=torch.zeros(self.env.num_envs,self._resolution[0],self._resolution[1],3)
        if self._read_images:
            self._sensor_image=torch.zeros(self.env.num_envs,self._resolution[0],self._resolution[1],3)

    def reset(self, envs_idx: list[int] | None = None):
        super().reset(envs_idx)
        if not self.enabled:
            return
        if self._read_rays:
            self._sensor_cloud[self.envs_idx]=0.0
        if self._read_images:
            self._sensor_image[self.envs_idx]=0.0

    def step(self):
        super().step()
        if not self.enabled:
            return
        
        if self._last_reading_timestamp is None:
            self._last_reading_timestamp=self.env.scene.cur_t
        elif self.env.scene.cur_t-self._last_reading_timestamp>self._sensor_read_interval:
            if self._read_rays:
                self._sensor_cloud[:]=self._spherical_raycaster_sensor.read()
            if self._read_images:
                self._sensor_cloud[:]=self._spherical_raycaster_sensor.read_image()
                
            self._last_reading_timestamp=self.env.scene.cur_t
