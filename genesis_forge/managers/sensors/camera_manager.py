from __future__ import annotations

import numpy as np
import torch
import genesis as gs
from genesis.utils.geom import quat_to_R, euler_to_quat

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.sensors.base_sensor_manager import BaseSensorManager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity

class CameraManager(BaseSensorManager):
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
        model : str
            Specifies the camera model. Options are 'pinhole' or 'thinlens'.
        res : tuple of int, shape (2,)
            The resolution of the camera, specified as a tuple (width, height).
        fov : float
            The vertical field of view of the camera in degrees.
        aperture : float
            The aperture size of the camera, controlling depth of field.
        focus_dist : float | None
            The focus distance of the camera. If None, it will be auto-computed using `pos` and `lookat`.
        GUI : bool
            Whether to display the camera's rendered image in a separate GUI window.
        spp : int, optional
            Samples per pixel. Only available when using RayTracer renderer. Defaults to 256.
        denoise : bool
            Whether to denoise the camera's rendered image. Only available when using the RayTracer renderer. Defaults
            to True on Linux, otherwise False. If OptiX denoiser is not available in your platform, consider enabling
            the OIDN denoiser option when building the RayTracer.
        near : float
            Distance from camera center to near plane in meters.
            Only available when using rasterizer in Rasterizer and BatchRender renderer. Defaults to 0.1.
        far : float
            Distance from camera center to far plane in meters.
            Only available when using rasterizer in Rasterizer and BatchRender renderer. Defaults to 20.0.
        env_idx : int, optional
            The specific environment index to bind to the camera. This option must be specified if and only if a
            non-batched renderer is being used. If provided, only this environment will be taken into account when
            following a rigid entity via 'follow_entity' and when being attached to some rigid link via 'attach'. Note
            that this option is unrelated to which environment is being rendering on the scene. Default to None for
            batched renderers (ie BatchRender), 'rendered_envs_idx[0]' otherwise (ie Raytracer or Rasterizer).
        debug_camera : bool
            Whether to use the debug camera. It enables to create cameras that can used to monitor / debug the
            simulation without being part of the "sensors". Their output is rendered by the usual simple Rasterizer
            systematically, no matter if BatchRender and RayTracer is enabled. This way, it is possible to record the
            simulation with arbitrary resolution and camera pose, without interfering with what robots can perceive
            from their environment. Defaults to False.
        render_on_demand: bool
            where to only render the camera on the 'get_(camera_type)_img' function call
        rgb: bool
            whether to render RGB images
        depth: bool
            whether to render depth images
        segmentation: bool
            whether to render segmentation images
        normal: bool
            whether to render Normal images
    """
    #TODO: write proper example

    def __init__(
        self,
        env: GenesisEnv,
        sensor_name,
        link_name: str,
        entity_attr: RigidEntity = "robot",
        pos_offset: list[float] = [0,0,0],
        euler_offset: list[float] = [0,0,0],
        delay: float = 0.0,
        draw_debug: bool = False,
        read_frequency: float= 15,
        model: str = 'pinhole',
        res: tuple[int, int] = (640, 480),
        fov: float = 45.0,
        aperture: float = 0.0,
        focus_dist: float | None = None,
        spp: int = 256,
        denoise: bool = True,
        near: float = 0.1,
        far: float = 20.0,
        env_idx: int | None = None,
        debug_camera: bool = False,
        render_on_demand=False,
        rgb: bool =True,
        depth: bool =False,
        segmentation: bool=False,
        normal: bool=False

    ):
        super().__init__(            
            env=env,
            sensor_type="camera_sensor",
            sensor_name=sensor_name,
            link_name=link_name,
            entity_attr=entity_attr,
            pos_offset=pos_offset,
            euler_offset=euler_offset,
            delay=delay,
            draw_debug=draw_debug,
            read_freuency=read_frequency)
        
        self._render_on_demand=render_on_demand
        self._render_rgb=rgb
        self._render_depth=depth
        self._render_segmentation=segmentation
        self._render_normal=normal
        
        self._camera_resolution=res
        self._camera_sensors=[]
        for env_idx in range(self.env.num_envs):
            self._camera_sensors.append(self.env.scene.add_camera(
                    model=model,
                    res=res,
                    fov=fov,
                    aperture=aperture,
                    focus_dist=focus_dist,
                    GUI=self._draw_debug,
                    spp=spp,
                    denoise=denoise,
                    near=near,
                    far=far,
                    env_idx=env_idx,
                    debug=debug_camera,
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
    def get_rgb_img(self) -> torch.Tensor:
        """
        RGB image rendered by the camera sensor 

        Returns:
            The RGB image shape is (n_envs,cam_res_width,cam_res_height, 3)
        """
        if not self._render_on_demand and self._render_rgb:
            return self._sensor_reading_rgb
        else:
            return self._render_all[0]
    
    def get_depth_img(self) -> torch.Tensor:
        """
        depth image rendered by the camera sensor 

        Returns:
            The depth image shape is (n_envs,cam_res_width,cam_res_height, 1)
        """
        if not self._render_on_demand and self._render_depth:
            return self._sensor_reading_depth
        else:
            return self._render_all()[1]

    def get_sgementation_img(self) -> torch.Tensor:
        """
        segmentation image rendered by the camera sensor 

        Returns:
            The segmentation image shape is (n_envs,cam_res_width,cam_res_height, 1)
        """
        if not self._render_on_demand and self._render_segmentation:
            return self._sensor_reading_segmentation
        else:
            return self._render_all()[2]
    
    def get_normal_img(self) -> torch.Tensor:
        """
        normal image rendered by the camera sensor 

        Returns:
            The normal image shape is (n_envs,cam_res_width,cam_res_height, 3)
        """
        if not self._render_on_demand and self._render_normal:
            return self._sensor_reading_normal
        else:
            return self._render_all()[3]

    """
    Lifecycle Operations
    """

    def build(self):
        """Initialize link indices and buffers."""
        super().build()
        T=np.eye(4)
        T[:3,:3]=quat_to_R(euler_to_quat(np.array(self._euler_offset)))
        T[:3,3]=self._pos_offset
        for camera_sensor in self._camera_sensors:
            camera_sensor.attach(self._link,offset_T=T)
        
        if not self._render_on_demand:
            if self._render_rgb:
                self._sensor_reading_rgb=torch.zeros(self.env.num_envs,
                                                     self._camera_resolution[0],
                                                     self._camera_resolution[1],3)        
            if self._render_depth:
                self._sensor_reading_depth=torch.zeros(self.env.num_envs,
                                                       self._camera_resolution[0],
                                                       self._camera_resolution[1],1)
            if self._render_segmentation:
                self._sensor_reading_segmentation=torch.zeros(self.env.num_envs,
                                                              self._camera_resolution[0],
                                                              self._camera_resolution[1],1)
            if self._render_normal:
                self._sensor_reading_normal=torch.zeros(self.env.num_envs,
                                                        self._camera_resolution[0],
                                                        self._camera_resolution[1],3)

    def reset(self, envs_idx: list[int] | None = None):
        super().reset(envs_idx)
        if not self.enabled:
            return
        for camera_sensor in self._camera_sensors:
            camera_sensor.move_to_attach()
        if not self._render_on_demand:
            if self._render_rgb:
                self._sensor_reading_rgb[self.envs_idx]=0.0
            if self._render_depth:
                self._sensor_reading_depth[self.envs_idx]=0.0
            if self._render_segmentation:
                self._sensor_reading_segmentation[self.envs_idx]=0.0
            if self._render_normal:
                self._sensor_reading_normal[self.envs_idx]=0.0

    def step(self):
        super().step()
        if not self.enabled:
            return
        if self._last_reading_timestamp is None:
            self._last_reading_timestamp=self.env.scene.cur_t
        elif self.env.scene.cur_t-self._last_reading_timestamp>self._sensor_read_interval:
            self._last_reading_timestamp=self.env.scene.cur_t
            if not self._render_on_demand:
                rgb,depth,segmentation,normal=self._render_all()
                if self._render_rgb:
                    self._sensor_reading_rgb=rgb
                if self._render_depth:
                    self._sensor_reading_depth=depth
                if self._render_segmentation:
                    self._sensor_reading_segmentation=segmentation
                if self._render_normal:
                    self._sensor_reading_normal=normal
    
    def _render_all(self):
        rgb_all=None
        depth_all=None
        segmentation_all=None
        normal_all=None
        for camera_sensor in self._camera_sensors:
            rgb,depth,segmentation,normal=camera_sensor.render(rgb=self._render_rgb,
                                                            depth=self._render_depth,
                                                            segmentation=self._render_segmentation,
                                                            normal=self._render_normal)
            if self._render_rgb:
                if rgb_all is None:
                    rgb_all=rgb
                else:
                    rgb_all=np.concat([rgb_all,rgb])
            if self._render_depth:
                if depth_all is None:
                    depth_all=depth
                else:
                    depth_all=np.concat([depth_all,depth])
            if self._render_segmentation:
                if segmentation_all is None:
                    segmentation_all=segmentation
                else:
                    segmentation_all=np.concat([segmentation_all,segmentation])
            if self._render_normal:
                if normal_all is None:
                    normal_all=normal
                else:
                    normal_all=np.concat([normal_all,normal])
        return (
            torch.tensor(rgb_all) if self._render_rgb else None,
            torch.tensor(depth_all) if self._render_depth else None,
            torch.tensor(segmentation_all) if self._render_segmentation else None,
            torch.tensor(normal_all) if self._render_normal else None
        )
            
                
