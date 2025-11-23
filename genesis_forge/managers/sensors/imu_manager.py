from __future__ import annotations

import torch
import genesis as gs

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.sensors.base_sensor_manager import BaseSensorManager
from typing import TYPE_CHECKING,Sequence

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity

class ImuManager(BaseSensorManager):
    """
    Inertal Measurenment Unit(IMU) sensor used to measure the lin_acc and ang_vel, it is attached to an entity's link in the environment.

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
        queue_length: The length of the IMU reading queue buffer
        acc_resolution : float, optional
            The measurement resolution of the accelerometer (smallest increment of change in the sensor reading).
            Default is 0.0, which means no quantization is applied.
        acc_axes_skew : float | tuple[float, float, float] | Sequence[float]
            Accelerometer axes alignment as a 3x3 rotation matrix, where diagonal elements represent alignment (0.0 to 1.0)
            for each axis, and off-diagonal elements account for cross-axis misalignment effects.
            - If a scalar is provided (float), all off-diagonal elements are set to the scalar value.
            - If a 3-element vector is provided (tuple[float, float, float]), off-diagonal elements are set.
            - If a full 3x3 matrix is provided, it is used directly.
        acc_bias : tuple[float, float, float]
            The constant additive bias for each axis of the accelerometer.
        acc_noise : tuple[float, float, float]
            The standard deviation of the white noise for each axis of the accelerometer.
        acc_random_walk : tuple[float, float, float]
            The standard deviation of the random walk, which acts as accumulated bias drift.
        gyro_resolution : float, optional
            The measurement resolution of the gyroscope (smallest increment of change in the sensor reading).
            Default is 0.0, which means no quantization is applied.
        gyro_axes_skew : float | tuple[float, float, float] | Sequence[float]
            Gyroscope axes alignment as a 3x3 rotation matrix, similar to `acc_axes_skew`.
        gyro_bias : tuple[float, float, float]
            The constant additive bias for each axis of the gyroscope.
        gyro_noise : tuple[float, float, float]
            The standard deviation of the white noise for each axis of the gyroscope.
        gyro_random_walk : tuple[float, float, float]
            The standard deviation of the bias drift for each axis of the gyroscope.
        debug_acc_color : float, optional
            The rgba color of the debug acceleration arrow. Defaults to (0.0, 1.0, 1.0, 0.5).
        debug_acc_scale: float, optional
            The scale factor for the debug acceleration arrow. Defaults to 0.01.
        debug_gyro_color : float, optional
            The rgba color of the debug gyroscope arrow. Defaults to (1.0, 1.0, 0.0, 0.5).
        debug_gyro_scale: float, optional
            The scale factor for the debug gyroscope arrow. Defaults to 0.01.
     
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
        delay: float=0.0,
        draw_debug: bool = False,
        read_frequency: float=100,
        queue_length: int = 25,
        acc_resolution: float = 0.0,
        acc_axes_skew: float | tuple[float, float, float] | Sequence[float] = 0.0,
        acc_bias: tuple[float, float, float] = (0.0, 0.0, 0.0),
        acc_noise: tuple[float, float, float] = (0.0, 0.0, 0.0),
        acc_random_walk: tuple[float, float, float] = (0.0, 0.0, 0.0),
        gyro_resolution: float = 0.0,
        gyro_axes_skew: float | tuple[float, float, float] | Sequence[float] = 0.0,
        gyro_bias: tuple[float, float, float] = (0.0, 0.0, 0.0),
        gyro_noise: tuple[float, float, float] = (0.0, 0.0, 0.0),
        gyro_random_walk: tuple[float, float, float] = (0.0, 0.0, 0.0),
        debug_acc_color: tuple[float, float, float, float] = (0.0, 1.0, 1.0, 0.5),
        debug_acc_scale: float = 0.01,
        debug_gyro_color: tuple[float, float, float, float] = (1.0, 1.0, 0.0, 0.5),
        debug_gyro_scale: float = 0.01,
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

        self._queue_length = queue_length

        scene=env.scene
        self._imu_sensor = scene.add_sensor(
            gs.sensors.IMU(
                **self.base_sensor_args,
                acc_resolution = acc_resolution,
                acc_axes_skew = acc_axes_skew,
                acc_bias = acc_bias,
                acc_noise = acc_noise,
                acc_random_walk = acc_random_walk,
                gyro_resolution = gyro_resolution,
                gyro_axes_skew = gyro_axes_skew,
                gyro_bias = gyro_bias,
                gyro_noise = gyro_noise,
                gyro_random_walk = gyro_random_walk,
                debug_acc_color = debug_acc_color,
                debug_acc_scale = debug_acc_scale,
                debug_gyro_color = debug_gyro_color,
                debug_gyro_scale = debug_gyro_scale,
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
    def get_lin_acceleration(self) -> torch.Tensor:
        """
        Get the linear acceleration of the imu 

        Returns:
            The linear acceleration of the imu. Shape is (n_envs, 3)
        """
        return self._imu_reading[:,0,:]
    
    def get_ang_velocity(self) -> torch.Tensor:
        """
        Get the angular velocity of the imu 

        Returns:
            The angular velocity of the imu. Shape is (n_envs, 3)
        """
        return self._imu_reading[:,1,:]

    def get_lin_acceleration_queue(self) -> torch.Tensor:
        """
        Get the linear acceleration buffer of the imu 

        Returns:
            The linear acceleration buffer of the imu. Shape is (n_envs, 3)
        """
        return self._imu_queue[:,:,0,:]
    
    def get_ang_velocity_queue(self) -> torch.Tensor:
        """
        Get the angular velocity buffer of the imu 

        Returns:
            The angular velocity buffer of the imu. Shape is (n_envs, 3)
        """
        return self._imu_queue[:,:,1,:]

    """
    Lifecycle Operations
    """

    def build(self):
        """Initialize buffers."""
        super().build()

        self._imu_reading=torch.zeros(self.num_envs,2,3)
        self._imu_queue=torch.zeros(self.queue_length,self.num_envs,2,3)

    def reset(self, envs_idx: list[int] | None = None):
        super().reset(envs_idx)
        self._imu_reading[self.envs_idx,:,:]=0.0
        self._imu_queue[:,self.envs_idx,:,:]=0.0

        
    def step(self):
        super().step()
        imu_data=self._imu_sensor.read()
        self._imu_reading[:,0,:]=imu_data[0]
        self._imu_reading[:,1,:]=imu_data[1]
        self._enqueue(self._imu_reading)

    """
    Internal Implementation
    """
    def _enqueue(self,imu_reading: torch.Tensor):
        if self._imu_queue.size(0) == self._queue_length:
            self._imu_queue = self._imu_queue[1:]  
            self._imu_queue = torch.cat((self._imu_queue, imu_reading.unsqueeze(0)))  
        else:
            self._imu_queue = torch.cat((self._imu_queue, imu_reading.unsqueeze(0)))

    def __repr__(self):
        attrs = [f"link_name={self._link_name}"]
        if self._entity_attr:
            attrs.append(f"entity_attr={self._entity_attr}")
        if self._pos_offset:
            attrs.append(f"pos_offset={self._pos_offset}")
        if self._euler_offset:
            attrs.append(f"pos_offset={self._euler_offset}")
        if self._queue_length:
            attrs.append(f"queue_length={self._queue_length}")
        attrs_str = ", ".join(attrs)
        return f"{self.__class__.__name__}({attrs_str})"
