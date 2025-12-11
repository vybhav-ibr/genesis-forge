import re
import torch
import numpy as np
from gymnasium import spaces
import genesis as gs
from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.actuator import ActuatorManager
from genesis_forge.managers.base import BaseManager

deprecated_arg_names = [
    "joint_names",
    "default_pos",
    "pd_kp",
    "pd_kv",
    "max_force",
    "damping",
    "stiffness",
    "frictionloss",
    "noise_scale",
]


class BaseActionManager(BaseManager):
    """
    Base for managers that handle actions.

    Args:
        env: The environment to manage the DOF actuators for.
        actuator_manager: The actuator manager which is used to setup and control the DOF joints.
        actuator_filter: Which joints of the actuator manager that this action manager will control.
                   These can be full names or regular expressions.
        delay_step: The number of steps to delay the actions for.
                    This is an easy way to emulate the latency in the system.
    """

    def __init__(
        self,
        env: GenesisEnv,
        actuator_manager: ActuatorManager | None = None,
        actuator_filter: list[str] | str = ".*",
        delay_step: int = 0,
        **kwargs,
    ):
        super().__init__(env, type="action")
        self._raw_actions = None
        self._actions = None
        self._delay_step = delay_step
        self._action_delay_buffer = []
        self._actuator_manager = actuator_manager
        self._actuator_filter = (
            [actuator_filter] if isinstance(actuator_filter, str) else actuator_filter
        )
        self._dofs: dict[int, str] = {}
        self._actuator_dof_filter: torch.Tensor = None

        # Deprecated actuator parameters
        deprecated_actuator_args = {
            key: kwargs[key] for key in deprecated_arg_names if key in kwargs
        }
        if len(deprecated_actuator_args) > 0:
            dep_list = ", ".join(deprecated_actuator_args.keys())
            if self._actuator_manager is not None:
                raise ValueError(
                    f"Cannot set both actuator_manager and deprecated actuator parameters: {dep_list}"
                )
            print(
                f"Actuator arguments are deprecated in the action manager, instead define an ActuatorManager ({dep_list})"
            )
            self._actuator_manager = ActuatorManager(
                env,
                joint_names=kwargs.get("joint_names", ".*"),
                default_pos=kwargs.get("default_pos", {".*": 0.0}),
                kp=kwargs.get("pd_kp", None),
                kv=kwargs.get("pd_kv", None),
                max_force=kwargs.get("max_force", None),
                damping=kwargs.get("damping", None),
                stiffness=kwargs.get("stiffness", None),
                frictionloss=kwargs.get("frictionloss", None),
                default_noise_scale=kwargs.get("noise_scale", 0.0),
            )
        if self._actuator_manager is None:
            raise ValueError("No ActuatorManager provided.")

    """
    Properties
    """

    @property
    def actuators(self) -> ActuatorManager:
        """
        Get the actuator manager.
        """
        return self._actuator_manager

    @property
    def num_actions(self) -> int:
        """
        Get the number of actions.
        """
        return len(self.dofs_idx)

    @property
    def dofs_idx(self) -> list[int]:
        """
        Get the indices of the DOFs that this action manager controls.
        """
        return list[int](self._dofs.values())

    @property
    def dofs(self) -> dict[str, int]:
        """
        Get a dictionary of the DOF names and their indices
        """
        return self._dofs

    @property
    def actuator_dof_filter(self) -> torch.Tensor:
        """
        An index filer for actuator DOF values.
        """
        return self._actuator_dof_filter

    @property
    def action_space(self) -> tuple[float, float]:
        """
        Returns the actions space for the environment, based on the number of DOFs defined in this action manager.
        """
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_actions,),
            dtype=np.float32,
        )

    @property
    def actions(self) -> torch.Tensor:
        """
        The actions for for the current step.
        """
        if self._actions is None:
            return torch.zeros((self.env.num_envs, self.num_actions))
        return self._actions

    @property
    def raw_actions(self) -> torch.Tensor:
        """
        The actions received from the policy, before being converted.
        """
        if self._raw_actions is None:
            return torch.zeros((self.env.num_envs, self.num_actions))
        return self._raw_actions

    """
    Lifecycle Operations
    """

    def build(self):
        """
        Builds the manager and initialized all the buffers.
        """
        # Filter the actuator DOFs that this action manager controls
        actuator_dofs = self._actuator_manager.dofs
        index_filter = []
        for filter in self._actuator_filter:
            for index, (name, dof_idx) in enumerate[tuple[str, int]](
                actuator_dofs.items()
            ):
                if name == filter or re.match(f"^{filter}$", name):
                    self._dofs[name] = dof_idx
                    index_filter.append(index)
        self._actuator_dof_filter = torch.tensor(
            index_filter, device=gs.device, dtype=gs.tc_int
        )

    def step(self, actions: torch.Tensor) -> None:
        """
        Handle the received actions.
        """
        # Action delay buffer
        if self._delay_step > 0:
            self._action_delay_buffer.insert(0, actions)
            actions = self._action_delay_buffer.pop()

        # Copy the actions into the manager buffer
        self._raw_actions = actions
        if self._actions is None:
            self._actions = torch.empty_like(actions, device=gs.device)
        self._actions[:] = self._raw_actions[:]
        return self._actions

    def reset(self, envs_idx: list[int] | None):
        """Reset environments."""
        if (
            self._delay_step > 0
            and len(self._action_delay_buffer) < self._delay_step
            and self.num_actions > 0
        ):
            while len(self._action_delay_buffer) < self._delay_step:
                self._action_delay_buffer.append(
                    torch.zeros((self.env.num_envs, self.num_actions), device=gs.device)
                )

    def get_actions(self) -> torch.Tensor:
        """
        Get the current actions for the environments.
        """
        if self._actions is None:
            return torch.zeros((self.env.num_envs, self.num_actions))
        return self._actions
