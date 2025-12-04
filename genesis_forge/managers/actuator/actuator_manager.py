from __future__ import annotations
import re
import torch
import genesis as gs
from typing import Literal, TypedDict, TYPE_CHECKING, Any, Union, Optional, List, Dict
from genesis_forge.genesis_env import GenesisEnv,EnvMode
from genesis_forge.managers.base import BaseManager
from genesis_forge.values import ensure_dof_pattern
from .noisy_value import NoisyValue

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity, HybridEntity, DroneEntity

ValueName = Literal[
    "kp",
    "kv",
    "default_pos",
    "force_min",
    "force_max",
    "damping",
    "stiffness",
    "frictionloss",
    "armature",
]


class BufferItem(TypedDict):
    buffer: torch.Tensor
    """A buffer of values for each DOF index."""

    noise: torch.Tensor
    """The noise scale for each index of the buffer."""

    noise_buffer: torch.Tensor
    """A pre-allocated buffer that will be filled with random noise values at each reset."""

    output_buffer: torch.Tensor
    """A pre-allocated buffer that will be filled with the values at each step."""

    has_noise: bool
    """Does this value have noise associated with it?"""

    has_been_set: bool
    """Has this value been set to the actuator at least once?"""


ValueBuffers = dict[ValueName, BufferItem]

class ActuatorConfig(TypedDict):
    env: GenesisEnv
    joint_names: Union[List[str], str]
    default_pos: Union[float, NoisyValue, Dict[str, float]]
    control_type: Union[str, NoisyValue, Dict[str, str]]
    kp: Union[float, NoisyValue, Dict[str, float]]
    kv: Union[float, NoisyValue, Dict[str, float]]
    dofs_limit: Union[Dict[str, tuple[float, float]]]
    max_force: Union[float, NoisyValue, tuple, Dict[str, Union[float, tuple]]]
    damping: Union[float, NoisyValue, Dict[str, float]]
    stiffness: Union[float, NoisyValue, Dict[str, float]]
    frictionloss: Union[float, NoisyValue, Dict[str, float]]
    armature: Union[float, NoisyValue, Dict[str, float]]
    default_noise_scale: float
    entity_attr: Optional[str]

class ActuatorManager(BaseManager):
    """
    Configures and manages the actuators of your robot.    
    You can define values for all actuators, or target specific joints by name or name pattern (see example).
    To add some domain randomization, you can define the values as `NoisyValue` objects, which will apply random noise at each reset.

    Args:
        actuator_config: A TypedDict containing all actuator configuration parameters:
            - env: The GenesisEnv instance.
            - joint_names: The joint names to manage (string or list of strings, supports regex). 
              Defaults to ".*".
            - default_pos: The default DOF positions. The DOF joints will be set to these positions on reset 
              (float, NoisyValue, or dict). Defaults to {".*": 0.0}.
            - control_type: Control type for joints ("position", "velocity", "force"). 
              Defaults to {".*": "position"}.
            - kp: The positional gain values (float, NoisyValue, or dict). Defaults to None.
            - kv: The velocity gain values (float, NoisyValue, or dict). Defaults to None.
            - dofs_limit: The DOF limits (dict of tuples). Defaults to None.
            - max_force: Define the maximum actuator force. Either as a single value or a tuple range 
              (float, NoisyValue, tuple, or dict). Defaults to None.
            - damping: The damping values (float, NoisyValue, or dict). Defaults to None.
            - stiffness: The stiffness values (float, NoisyValue, or dict). Defaults to None.
            - frictionloss: The frictionloss values (float, NoisyValue, or dict). Defaults to None.
            - armature: The armature values (float, NoisyValue, or dict). Defaults to None.
            - default_noise_scale: (deprecated) This noise scale will be applied to all actuator values. 
              Use `NoisyValue` instead. Defaults to 0.0.
            - entity_attr: The attribute of the environment to get the robot from (str or None). 
              Defaults to None.

    Example::

        class MyEnv(GenesisEnv):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                actuator_config = {
                    "env": self,
                    "joint_names": ".*",
                    "default_pos": {
                        "Leg[1-2]_Hip": -1.0,
                        "Leg[3-4]_Hip": 1.0,
                        "Leg[1-4]_Femur": 0.5,
                        "Leg[1-4]_Tibia": 0.6,
                    },
                    "kp": {
                        ".*_Hip": NoisyValue(50, 0.02),
                        ".*_Femur": NoisyValue(30, 0.01),
                        ".*_Tibia": NoisyValue(30, 0.01),
                    },
                    "kv": 0.5,
                    "dofs_limit: {"joint1":(-1.57, 1.57), "joint2":(-1.57, 1.57),}
                    "max_force": 8.0,
                }
                
                self.actuator_manager = ActuatorManager(actuator_config)

    """

    def __init__(
        self,
        actuator_config: ActuatorConfig
    ) -> None:
        env = actuator_config.get("env")
        self.env_mode=env.env_mode
        entity_attr = actuator_config.get("entity_attr", None)
        
        super().__init__(env, type="actuator")
        self._dofs: dict[str, int] = {}
        
        # Initialize robot entity
        if entity_attr is not None and self.env_mode!=EnvMode.DEPLOY:
            self._robot: RigidEntity | HybridEntity | DroneEntity = getattr(env, entity_attr)
            self._actuator_config = actuator_config
        else:
            # Standalone mode - robot properties will be extracted from config
            # self._robot = None
            self._actuator_config = actuator_config
        
        if hasattr(env, "scene"):
            self._batch_dofs_enabled = (
                env.scene.rigid_options.batch_dofs_info
                and env.scene.rigid_options.batch_links_info
            )
        else:
            self._batch_dofs_enabled=False

        # Convert all actuator parameter configurations to DOF pattern dictionaries
        # These allow per-joint or regex-based pattern configuration
        self._default_pos_cfg = ensure_dof_pattern(actuator_config.get("default_pos", {".*": 0.0}))
        self._control_type_cfg = ensure_dof_pattern(actuator_config.get("control_type", {".*": "position"}))
        self._kp_cfg = ensure_dof_pattern(actuator_config.get("kp", None))
        self._kv_cfg = ensure_dof_pattern(actuator_config.get("kv", None))
        self._max_force_cfg = ensure_dof_pattern(actuator_config.get("max_force", None))
        self._damping_cfg = ensure_dof_pattern(actuator_config.get("damping", None))
        self._stiffness_cfg = ensure_dof_pattern(actuator_config.get("stiffness", None))
        self._frictionloss_cfg = ensure_dof_pattern(actuator_config.get("frictionloss", None))
        self._armature_cfg = ensure_dof_pattern(actuator_config.get("armature", None))
        self._default_noise_scale = actuator_config.get("default_noise_scale", 0.0)
        
        # Initialize action and state tracking lists
        self._pos_actions = []
        self._vel_actions = []
        self._force_actions = []
        self._pos_states = []
        self._vel_states = []
        self._force_states = []

        # Initialize value buffers for all actuator parameters
        # These will be populated during the build() phase
        self._values: ValueBuffers = {
            "default_pos": None,
            "force_min": None,
            "force_max": None,
            "kp": None,
            "kv": None,
            "damping": None,
            "stiffness": None,
            "frictionloss": None,
            "armature": None,
        }

        # Normalize joint names to list format
        joint_names = actuator_config.get("joint_names", ".*")
        if isinstance(joint_names, str):
            self._joint_name_cfg = [joint_names]
        elif isinstance(joint_names, list):
            self._joint_name_cfg = joint_names

        # print("joint_cfg:",self._joint_name_cfg)
        # exit(0)
        # Initialize DOF index and name lists categorized by control type
        self._pos_dofs_idx = []
        self._vel_dofs_idx = []
        self._force_dofs_idx = []
        self._pos_dofs_names = []
        self._vel_dofs_names = []
        self._force_dofs_names = []

    """
    Properties
    """

    @property
    def num_dofs(self) -> int:
        """
        Get the number of configured DOFs.
        """
        return len(self._dofs)

    @property
    def dofs_idx(self) -> list[int]:
        """
        Get the indices of the DOF that are enabled (via joint_names).
        """
        return list[int](self._dofs.values())
    
    @property
    def pos_dofs_idx(self) -> list[int]:
        """
        Get the indices of the DOF that are enabled (via joint_names).
        """
        return self._pos_dofs_idx
    
    @property
    def vel_dofs_idx(self) -> list[int]:
        """
        Get the indices of the DOF that are enabled (via joint_names).
        """
        return self._vel_dofs_idx
    
    @property
    def force_dofs_idx(self) -> list[int]:
        """
        Get the indices of the DOF that are enabled (via joint_names).
        """
        return self._force_dofs_idx

    @property
    def joint_names(self) -> list[str]:
        """
        Get the names of the joints that are enabled, in the order of the DOF indices.
        """
        return list[str](self._dofs.keys())
    
    @property
    def dofs_names(self) -> list[str]:
        """
        Get the names of the configured DOFs.
        """
        return list[str](self._dofs.keys())
    
    @property
    def pos_dofs_names(self) -> list[str]:
        """
        Get the names of the configured DOFs with position control.
        """
        return self._pos_dofs_names
    
    @property
    def vel_dofs_names(self) -> list[str]:
        """
        Get the names of the configured DOFs with position control.
        """
        return self._vel_dofs_names
    
    @property
    def force_dofs_names(self) -> list[str]:
        """
        Get the names of the configured DOFs with force control.
        """
        return self._force_dofs_names

    @property
    def default_dofs_pos(self) -> torch.Tensor:
        """
        Return the default DOF positions.
        """
        return self._values.get("default_pos", {}).get("buffer", None)
    
    @property
    def propeller_links_idx(self) -> list[int]:
        """
        Get the link_idxs of the propeller links 
        """
        return self._robot_propellers_link_idxs
    
    @property
    def num_propellers(self) -> int:
        """
        Get the number of propellers of the robot
        """
        return self._robot_num_propellers

    """
    Actuator handlers
    """

    def get_dofs_position(self, noise: float = 0.0):
        """
        Return the current position of the configured DOFs.
        This is a wrapper for `RigidEntity.get_dofs_position`.

        Args:
            noise: The maximum amount of random noise to add to the position values returned.
        """
        pos = self._robot.get_dofs_position(self.dofs_idx)
        if noise > 0.0:
            pos = self._add_random_noise(pos, noise)
        return pos

    def get_dofs_velocity(self, noise: float = 0.0, clip: tuple[float, float] = None):
        """
        Return the current velocity of the configured DOFs.
        This is a wrapper for `RigidEntity.get_dofs_velocity`.

        Args:
            noise: The maximum amount of random noise to add to the velocity values returned.
            clip: Clip the velocity returned.
        """
        vel = self._robot.get_dofs_velocity(self.dofs_idx)
        if noise > 0.0:
            vel = self._add_random_noise(vel, noise)
        if clip is not None:
            vel = vel.clamp(**clip)
        return vel

    def get_dofs_force(self, noise: float = 0.0, clip_to_max_force: bool = False):
        """
        Return the force experienced by the configured DOFs.
        This is a wrapper for `RigidEntity.get_dofs_force`.

        Args:
            noise: The maximum amount of random noise to add to the force values returned.
            clip_to_max_force: Clip the force returned to the maximum force of the actuators.

        Returns:
            The force experienced by the enabled DOFs.
        """
        force = self._robot.get_dofs_force(self.dofs_idx)
        if noise > 0.0:
            force = self._add_random_noise(force, noise)
        if clip_to_max_force:
            [lower, upper] = self._robot.get_dofs_force_range(self.dofs_idx)
            force = force.clamp(lower, upper)
        return force

    def get_dofs_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the limits of the configured DOFs.
        This is a wrapper for `RigidEntity.get_dofs_limit`.

        Returns:
            A tuple of two tensors, the first is the lower limits and the second is the upper limits.
            Each tensor is of shape (num_envs, num_dofs).
        """
        if not hasattr(self,"_robot"):
            # print("dof_limits_device:",self._robot_dof_limits[0].device)
            return self._robot_dof_limits
        return self._robot.get_dofs_limit(self.dofs_idx)    
    
    def get_dofs_force_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the force limits of the configured DOFs.
        This is a wrapper for `RigidEntity.get_dofs_force_range`.

        Returns:
            A tuple of two tensors, the first is the lower limits and the second is the upper limits.
            Each tensor is of shape (num_envs, num_dofs).
        """
        if not hasattr(self,"_robot"):
            return self._robot_dof_force_limits
        return self._robot.get_dofs_force_range(self.dofs_idx)

    def set_dofs_position(self, position: torch.Tensor):
        """
        Set the position of the configured DOFs.
        This is a wrapper for `RigidEntity.set_dofs_position`.

        Args:
            position: The position to set the DOFs to. The indices of this tensor should match the configured DOFs
                      (see: `dofs_names` and `dofs_idx` properties).
        """
        self._robot.set_dofs_position(position, self.dofs_idx)

    def control_dofs_position(self, position: torch.Tensor):
        """
        Control the position of the configured DOFs.
        This is a wrapper for `RigidEntity.control_dofs_position`.

        Args:
            position: The position to set the DOFs to. The indices of this tensor should match the configured DOFs
                      (see: `dofs_names` and `dofs_idx` properties).
        """
        if self.pos_dofs_idx is not None and len(self.pos_dofs_idx) > 0:
            self._robot.control_dofs_position(position, self.pos_dofs_idx)
        
    def set_dofs_velocity(self, velocity: torch.Tensor):
        """
        Set the velocity of the configured DOFs.
        This is a wrapper for `RigidEntity.set_dofs_velocity`.

        Args:
            position: The position to set the DOFs to. The indices of this tensor should match the configured DOFs
                      (see: `dofs_names` and `dofs_idx` properties).
        """
        self._robot.set_dofs_velocity(velocity, self.dofs_idx)

    def control_dofs_velocity(self, velocity: torch.Tensor):
        """
        Control the velocity of the configured DOFs.
        This is a wrapper for `RigidEntity.control_dofs_velocity`.

        Args:
            position: The position to set the DOFs to. The indices of this tensor should match the configured DOFs
                      (see: `dofs_names` and `dofs_idx` properties).
        """
        if self.vel_dofs_idx is not None and len(self.vel_dofs_idx) > 0:
            self._robot.control_dofs_velocity(velocity, self.vel_dofs_idx)
            
    def set_dofs_force(self, force: torch.Tensor):
        """
        Set the force of the configured DOFs.
        This is a wrapper for `RigidEntity.set_dofs_force`.
        Args:
            force: The force to set the DOFs to. The indices of this tensor should match the configured DOFs
                      (see: `dofs_names` and `dofs_idx` properties).
        """
        self._robot.set_dofs_force(force, self.dofs_idx)

    def control_dofs_force(self, force: torch.Tensor):
        """
        Control the force of the configured DOFs.
        This is a wrapper for `RigidEntity.control_dofs_force`.

        Args:
            force: The force to set the DOFs to. The indices of this tensor should match the configured DOFs
                      (see: `dofs_names` and `dofs_idx` properties).
        """
        if self.force_dofs_idx is not None and len(self.force_dofs_idx) > 0:
            self._robot.control_dofs_force(force, self.force_dofs_idx)
            
    def set_propellels_rpm(self, rpm: torch.Tensor):
        """
        Set the propellers's rpm 
        
        Args:
            rpm: The rpm to set the propellers to. This function expects the rpm for all the propellers 
        """
        self._robot.set_propellels_rpm(rpm) 

    """
    Lifecycle operations
    """

    def build(self):
        """
        Builds the manager and initialized all the buffers.
        """
        # Find all configured joints by names/patterns
        if self.env_mode != EnvMode.DEPLOY:
            for joint in self._robot.joints:
                if joint.type != self.env.REVOLUTE_JOINT_TYPE and joint.type != self.env.PRISMATIC_JOINT_TYPE:
                    continue
                name = joint.name
                for pattern in self._joint_name_cfg:
                    if pattern == name or re.match(f"^{pattern}$", name):
                        self._dofs[name] = joint.dof_start
                        break
        else:
            for idx,joint_name in enumerate(self._joint_name_cfg):
                self._dofs[joint_name] = idx

        self._get_dof_idx_and_names()
        self._get_robot_configs()
        # If no configuration is provided, use zero positions for all DOFs.
        if self._default_pos_cfg is not None:
            self._fill_value_buffer("default_pos", self._default_pos_cfg)
        else:
            self._fill_value_buffer("default_pos", {".*": 0.0})

        # Max force
        # The value can either be a single float or a tuple range
        # First normalize them into two dicts: min and max
        if self._max_force_cfg is not None:
            # Normalize the max_force values into a min & max dict
            force_min = {}
            force_max = {}
            for pattern, value in self._max_force_cfg.items():
                if isinstance(value, list):
                    force_min[pattern] = value[0]
                    force_max[pattern] = value[1]
                elif isinstance(value, NoisyValue):
                    force_min[pattern] = NoisyValue(-value.value, value.noise)
                    force_max[pattern] = value
                else:
                    force_min[pattern] = -value
                    force_max[pattern] = value

            self._fill_value_buffer("force_min", force_min)
            self._fill_value_buffer("force_max", force_max)

        # Armature
        # If DOF batching is not enabled, print a warning and set the armature values for all environments without noise.
        if self._armature_cfg is not None:
            self._fill_value_buffer("armature", self._armature_cfg)
            if not self._batch_dofs_enabled:
                armature = self._values["armature"]
                if torch.any(armature["noise"] != 0.0):
                    print(
                        "WARNING: Armature randomization settings are only supported when 'batch_dofs_info' and 'batch_links_info' are True in RigidOptions."
                    )
                self._robot.set_dofs_armature(armature["buffer"], self.dofs_idx)

        # Other actuator values
        self._fill_value_buffer("kp", self._kp_cfg)
        self._fill_value_buffer("kv", self._kv_cfg)
        self._fill_value_buffer("damping", self._damping_cfg)
        self._fill_value_buffer("stiffness", self._stiffness_cfg)
        self._fill_value_buffer("frictionloss", self._frictionloss_cfg)

    def reset(
        self,
        envs_idx: list[int] = None,
    ):
        """Reset the DOF positions."""
        if not self.enabled:
            return
        if not hasattr(self,"_robot"):
            return
        dofs_idx = self.dofs_idx

        # Set actuator controller values
        if self._should_set_value("kp"):
            kp = self._get_value_buffer("kp", envs_idx)
            self._robot.set_dofs_kp(kp, dofs_idx, envs_idx)
            self._values["kp"]["has_been_set"] = True

        if self._should_set_value("kv"):
            kv = self._get_value_buffer("kv", envs_idx)
            self._robot.set_dofs_kv(kv, dofs_idx, envs_idx)
            self._values["kv"]["has_been_set"] = True

        if self._should_set_value("damping"):
            damping = self._get_value_buffer("damping", envs_idx)
            self._robot.set_dofs_damping(damping, dofs_idx, envs_idx)
            self._values["damping"]["has_been_set"] = True

        if self._should_set_value("stiffness"):
            stiffness = self._get_value_buffer("stiffness", envs_idx)
            self._robot.set_dofs_stiffness(stiffness, dofs_idx, envs_idx)
            self._values["stiffness"]["has_been_set"] = True

        if self._should_set_value("frictionloss"):
            frictionloss = self._get_value_buffer("frictionloss", envs_idx)
            self._robot.set_dofs_frictionloss(frictionloss, dofs_idx, envs_idx)
            self._values["frictionloss"]["has_been_set"] = True

        if self._should_set_value("armature") and self._batch_dofs_enabled:
            armature = self._get_value_buffer("armature", envs_idx)
            self._robot.set_dofs_armature(armature, dofs_idx, envs_idx)
            self._values["armature"]["has_been_set"] = True

        if self._should_set_value("force_min") or self._should_set_value("force_max"):
            force_min = self._get_value_buffer("force_min", envs_idx)
            force_max = self._get_value_buffer("force_max", envs_idx)
            self._robot.set_dofs_force_range(force_min, force_max, dofs_idx, envs_idx)
            self._values["force_min"]["has_been_set"] = True
            self._values["force_max"]["has_been_set"] = True

        # Reset DOF positions
        default_position = self._get_value_buffer("default_pos", envs_idx)
        self._robot.set_dofs_position(
            position=default_position,
            dofs_idx_local=dofs_idx,
            envs_idx=envs_idx,
        )

    """
    Internal methods
    """

    def _should_set_value(self, value_name: ValueName) -> bool:
        """
        Check if the actuator control value should.
        We don't want to set the value if we've already set it and there is no noise associated with it.
        """
        cfg = self._values[value_name]
        if cfg is None:
            return False
        if cfg["has_been_set"] and not cfg["has_noise"]:
            return False
        return True

    def _fill_value_buffer(
        self, value_name: ValueName, config: NoisyValue[float] | None
    ):
        """
        Given a ActuatorValue dict, loop over the entries, and set them to the appropriate value buffer DOF indices that match
        the DOF pattern.

        Args:
            name: The name of the value buffer to fill
            values: The DOF value to convert (for example: `{".*": 50}`).

        """
        num_dofs = len(self._dofs)
        is_idx_set = [False] * num_dofs
        dof_names = self._dofs.keys()
        print("dof_names:",list(dof_names))
        has_noise = False

        # Nothing to be done if the config is None
        if config is None:
            return

        # Initialize the buffers
        value_buffer = torch.zeros((num_dofs,), device=self.env.device, dtype=self.env.float_type)
        noise = torch.zeros((num_dofs,), device=self.env.device, dtype=self.env.float_type).fill_(
            self._default_noise_scale
        )
        noise_buffer = torch.zeros_like(value_buffer, device=self.env.device)
        output_buffer = torch.zeros_like(value_buffer, device=self.env.device)

        for pattern, value in config.items():
            found = False
            for i, name in enumerate[str](dof_names):
                if is_idx_set[i]:
                    continue
                if pattern == name or re.match(f"^{pattern}$", name):
                    found = True
                    is_idx_set[i] = True

                    if isinstance(value, NoisyValue):
                        noise[i] = value.noise
                        value_buffer[i] = value.value
                        has_noise = True
                    else:
                        value_buffer[i] = value
            if not found:
                raise RuntimeError(f"Joint DOF '{pattern}' not found.")

        # Expand the default postion buffer to the number of environments
        if value_name == "default_pos" or self._batch_dofs_enabled:
            value_buffer = value_buffer.unsqueeze(0).repeat(self.env.num_envs, 1)
            noise = noise.unsqueeze(0).repeat(self.env.num_envs, 1)
            noise_buffer = noise_buffer.unsqueeze(0).repeat(self.env.num_envs, 1)
            output_buffer = output_buffer.unsqueeze(0).repeat(self.env.num_envs, 1)

        # Expand the buffer to the number of environments
        self._values[value_name] = {
            "buffer": value_buffer,
            "noise": noise,
            "noise_buffer": noise_buffer,
            "output_buffer": output_buffer,
            "has_noise": has_noise,
            "has_been_set": False,
        }

    def _get_value_buffer(
        self, name: ValueName, envs_idx: list[int] | None = None
    ) -> torch.Tensor:
        """
        Get the value buffer tensor, with noise applied
        """
        output = self._values[name]["output_buffer"]
        noise = self._values[name]["noise"]
        noise_buffer = self._values[name]["noise_buffer"]

        output[:] = self._values[name]["buffer"]
        output += noise_buffer.uniform_(-1, 1) * noise
        if envs_idx is not None and output.ndim == 2:
            output = output[envs_idx]
        return output

    def _add_random_noise(
        self, values: torch.Tensor, noise_scale: float = 0.0
    ) -> torch.Tensor:
        """
        Add random noise to the tensor values
        """
        if noise_scale == 0.0:
            return values
        noise_value = torch.empty_like(values).uniform_(-1, 1) * noise_scale
        return values + noise_value
    
    def _get_dof_idx_and_names(self):
        for name, idx in self._dofs.items():
            control_type = None
            for pattern, value in self._control_type_cfg.items():
                if re.match(f"^{pattern}$", name):
                    control_type = value
                    break
            if control_type == "position":
                self._pos_dofs_idx.append(idx)
                self._pos_dofs_names.append(name)
            elif control_type == "velocity":
                self._vel_dofs_idx.append(idx)
                self._vel_dofs_names.append(name)
            elif control_type == "force":
                self._force_dofs_idx.append(idx)
                self._force_dofs_names.append(name)
                
    def _get_robot_configs(self):
         # Initialize robot-specific properties (propellers, DOF limits)
        if hasattr(self,"_robot"):
            # Get properties directly from the robot entity
            try:
                self._robot_propellers_link_idxs = self._robot.propellers_link_idxs 
                self._robot_num_propellers = self._robot.num_propellers
            except AttributeError:
                self._robot_propellers_link_idxs = None
                self._robot_num_propellers = None
        else:
            # Standalone mode: extract properties from config
            self._robot_propellers_link_idxs = self._actuator_config.get("propellers_link_idx")
            self._robot_num_propellers = self._actuator_config.get("num_propellers")
            
            # Convert DOF limits to tensors if provided
            dofs_limit = self._actuator_config.get("dofs_limit")
            if dofs_limit is not None:
                # dofs_limit is a dict with regex patterns as keys, mapping to (lower, upper) tuples
                # We need to parse it and create tensors for lower and upper limits
                lower_limits = []
                upper_limits = []
                
                for dof_name in self._dofs.keys():
                    # Match DOF name against patterns in dofs_limit
                    matched = False
                    for pattern, limits in dofs_limit.items():
                        if re.match(f"^{pattern}$", dof_name):
                            lower, upper = limits
                            lower_limits.append(lower)
                            upper_limits.append(upper)
                            matched = True
                            break
                    
                    if not matched:
                        raise ValueError(f"DOF '{dof_name}' does not match any pattern in dofs_limit configuration")
                
                self._robot_dof_limits = (
                    torch.tensor(lower_limits,device=self.env.device),
                    torch.tensor(upper_limits,device=self.env.device)
                )
            # Convert DOF force limits to tensors if provided
            dofs_force_limit = self._actuator_config.get("dofs_force_limit")
            if dofs_force_limit is not None:
                # dofs_force_limit is a dict with regex patterns as keys, mapping to (lower, upper) tuples
                lower_force_limits = []
                upper_force_limits = []
                
                for dof_name in self._dofs.keys():
                    # Match DOF name against patterns in dofs_force_limit
                    matched = False
                    for pattern, limits in dofs_force_limit.items():
                        if re.match(f"^{pattern}$", dof_name):
                            lower, upper = limits
                            lower_force_limits.append(lower)
                            upper_force_limits.append(upper)
                            matched = True
                            break
                    
                    if not matched:
                        raise ValueError(f"DOF '{dof_name}' does not match any pattern in dofs_force_limit configuration")
                
                self._robot_dof_force_limits = (
                    torch.tensor(lower_force_limits),
                    torch.tensor(upper_force_limits)
                )

