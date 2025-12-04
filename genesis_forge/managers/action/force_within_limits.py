from __future__ import annotations
import torch
from typing import Callable, TypedDict, Optional

from genesis_forge.genesis_env import GenesisEnv
from .force_action_manager import ForceActionManager, ForceActionConfig
from genesis_forge.managers.actuator import ActuatorManager


class ForceWithinLimitsActionConfig(TypedDict):
    env: GenesisEnv
    actuator_manager: Optional[ActuatorManager]
    action_handler: Optional[Callable[[torch.Tensor], None]]
    quiet_action_errors: bool
    delay_step: int


class ForceWithinLimitsActionManager(ForceActionManager):
    """
    This is similar to `ForceActionManager` but converts actions from the range -1.0 - 1.0 to DOF force within the limits of the actuators.

    Args:
        action_config: A ForceWithinLimitsActionConfig TypedDict containing all configuration parameters:
            - env: The environment to manage the DOF actuators for.
            - actuator_manager: The actuator manager which is used to setup and control the DOF joints.
            - action_handler: Optional custom action handler. Defaults to None.
            - quiet_action_errors: Whether to quiet action errors. Defaults to False.
            - delay_step: The number of steps to delay the actions for. This is an easy way to emulate 
              the latency in the system. Defaults to 0.

    Example::

        class MyEnv(GenesisManagedEnvironment):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def config(self):
                self.actuator_manager = ActuatorManager({
                    "env": self,
                    "joint_names": ".*",
                    "default_pos": {
                        # Hip joints
                        "Leg[1-2]_Hip": -1.0,
                        "Leg[3-4]_Hip": 1.0,
                        # Femur joints
                        "Leg[1-4]_Femur": 0.5,
                        # Tibia joints
                        "Leg[1-4]_Tibia": 0.6,
                    },
                    "kp": {".*": 50},
                    "kv": {".*": 0.5},
                    "max_force": {".*": 8.0},
                    "entity_attr": "robot"
                })
                self.action_manager = ForceWithinLimitsActionManager({
                    "env": self,
                    "actuator_manager": self.actuator_manager,
                })

    """

    def __init__(
        self,
        action_config: ForceWithinLimitsActionConfig | None = None,
        **kwargs,
    ):
        # Support both old and new API
        if action_config is None:
            # Old API: individual parameters
            action_config = {
                "env": kwargs.pop("env", None),
                "actuator_manager": kwargs.pop("actuator_manager", None),
                "action_handler": kwargs.pop("action_handler", None),
                "quiet_action_errors": kwargs.pop("quiet_action_errors", False),
                "delay_step": kwargs.pop("delay_step", 0),
            }
        
        # Create parent config with required fields
        parent_config: ForceActionConfig = {
            "env": action_config.get("env"),
            "actuator_manager": action_config.get("actuator_manager"),
            "scale": 1.0,  # Will be overridden in build()
            "clip": None,
            "action_handler": action_config.get("action_handler"),
            "quiet_action_errors": action_config.get("quiet_action_errors", False),
            "delay_step": action_config.get("delay_step", 0),
        }
        
        super().__init__(parent_config, **kwargs)

    """
    Lifecycle Operations
    """

    def build(self):
        """
        Builds the manager and initialized all the buffers.
        """
        lower, upper = self._actuator_manager.get_dofs_force_range()
        lower = lower.unsqueeze(0).expand(self.env.num_envs, -1)
        upper = upper.unsqueeze(0).expand(self.env.num_envs, -1)
        self._offset = (upper + lower) * 0.5
        self._scale = (upper - lower) * 0.5

    def handle_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Converts the actions to force commands, and send them to the DOF actuators.
        Override this function if you want to change the action handling logic.

        Args:
            actions: The incoming step actions to handle.

        Returns:
            The processed and handled actions.
        """
        # Convert the action from -1 to 1, to absolute position within the actuator limits
        actions.clamp_(-1.0, 1.0)
        self._actions = actions * self._scale

        # Set target positions
        self._actuator_manager.control_dofs_force(self._actions)

        return self._actions
