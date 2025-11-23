from __future__ import annotations

import torch
import genesis as gs

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.base import BaseManager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity

class BaseSensorManager(BaseManager):
    """
    Base sensor for handling a sensor.

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
        
     # ...other arguments here...
    """
    #TODO: write proper example

    def __init__(
        self,
        env: GenesisEnv,
        sensor_type: str,
        sensor_name: str,
        link_name: str,
        entity_attr: RigidEntity = "robot",
        pos_offset: list[float]=[0,0,0],
        euler_offset: list[float]=[0,0,0],
        delay: float=0.0,
        draw_debug: bool=False,
        read_freuency: float=20,
    ):
        super().__init__(env, type=sensor_type)
        self._sensor_name = sensor_name
        self._link_name = link_name
        self._entity_attr = entity_attr
        self._pos_offset=pos_offset
        self._euler_offset=euler_offset
        self._delay=delay
        self._draw_debug=draw_debug
        self._read_frequncy=read_freuency
        self._sensor_read_interval=1/self._read_frequncy
        
        # Get the link indices
        if self._link_name is None:
            self._link_name="base_link"
        self._entity,self._link = self._get_entity_and_link(
            self._entity_attr, self._link_name
        )
        
        self.base_sensor_args=dict(
            entity_idx=self._entity.idx,
            link_idx_loacl=self._link.idx_local,
            pos_offset=self._pos_offset,
            euler_offset=self._euler_offset
        )
        self._last_reading_timestamp=None

    """
    Properties
    """
    @property
    def sensor_name(self) -> torch.Tensor:
        """name of the sensor."""
        return self._sensor_name

    @property
    def link_name(self) -> torch.Tensor:
        """name of the sensor link."""
        return self._link_name
    
    @property
    def link_idx(self) -> torch.Tensor:
        """The link index for the sensor link."""
        return self._link.idx
    
    @property
    def local_link_idx(self) -> torch.Tensor:
        """The local link index for the sensor link."""
        return self._link.idx_local

    """
    Lifecycle Operations
    """

    def build(self):
        """Initialize link indices and buffers."""
        super().build()

    def reset(self, envs_idx: list[int] | None = None):
        super().reset(envs_idx)
        if not self.enabled:
            return
        if envs_idx is None:
            self.envs_idx = torch.arange(self.env.num_envs, device=gs.device)
        else:
            self.envs_idx=envs_idx

        
    def step(self):
        super().step()
        if not self.enabled:
            return
        
    def disable_sensor(self):
        self.enabled=False

    """
    Internal Implementation
    """

    def _get_entity_and_link(
        self, entity_attr: str, link_name: str
    ):
        """
        Find the link handle for the given link name and entity_name.

        Args:
            entity: The entity to find the links in.
            link: The link found in the entity

        Returns: Tuple of global and local link index tensors.
        """
        entity = self.env.__getattribute__(entity_attr)
        try:
            link=entity.get_link(link_name)
        except Exception as e:
            link=entity.links[0]

        return entity,link

    def __repr__(self):
        attrs = [f"sensor_name={self._sensor_name}",f"link_name={self._link_name}"]
        if self._entity_attr:
            attrs.append(f"entity_attr={self._entity_attr}")
        if self._pos_offset:
            attrs.append(f"pos_offset={self._pos_offset}")
        if self._euler_offset:
            attrs.append(f"pos_offset={self._euler_offset}")
        if self._delay:
            attrs.append(f"delay={self._delay}")
        if self._draw_debug:
            attrs.append(f"draw_debug={self._draw_debug}")
        attrs_str = ", ".join(attrs)
        return f"{self.__class__.__name__}({attrs_str})"
