import os
import glob
import torch
import pickle
import argparse
from importlib import metadata
import re
import types
import time
 
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from builtin_interfaces.msg import Time
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist,Pose

from genesis_forge.managers import ActuatorManager, PositionActionManager
from genesis_forge.wrappers import RslRlWrapper
from environment import Go2SimpleEnv

from genesis_forge.managers import ObservationManager

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib").startswith("1."):
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install install 'rsl-rl-lib>=2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

EXPERIMENT_NAME = "go2-simple"

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("-d", "--device", type=str, default="gpu")
parser.add_argument("-e", "--exp_name", type=str, default=EXPERIMENT_NAME)
args = parser.parse_args()


def get_latest_model(log_dir: str) -> str:
    """
    Get the last model from the log directory
    """
    model_checkpoints = glob.glob(os.path.join(log_dir, "model_*.pt"))
    if len(model_checkpoints) == 0:
        print(
            f"Warning: No model files found at '{log_dir}' (you might need to train more)."
        )
        exit(1)
    # Sort by the file with the highest number
    sorted_models = sorted(
        model_checkpoints,
        key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]),
    )
    return sorted_models[-1]

class RosInterface(Node):
    def __init__(self):
        super().__init__("ros_interface")
        self._ros_node = self
        self._ros_clock = Clock()
        
        self.env = None
        self.actuator_manager = None
        
        self._pos_actions = []
        self._vel_actions = []
        self._force_actions = []
        
        self._pos_state = []
        self._vel_state = []
        self._force_state = []
        
        self._robot_lin_vel = None
        self._robot_ang_vel = None
        self._robot_pos = None
        self._robot_quat = None

    def configure(self, env):
        """
        Configure the interface with environment details
        """
        self.env = env
        self.actuator_manager = env.actuator_manager
        
    def setup_ros(self):
        """
        Setup ROS publishers and subscribers. 
        Should be called after env.build() so that joint names are available.
        """
        # Initialize action buffers
        num_dofs = len(self.dofs_idx)
        self._pos_actions = torch.zeros(num_dofs)
        self._vel_actions = torch.zeros(num_dofs)
        self._force_actions = torch.zeros(num_dofs)

        # Setup subscribers and publishers
        self._setup_joint_action_publisher()
        self._setup_robot_twist_subscriber()
        self._setup_robot_pose_subscriber()
        self._setup_joint_state_subscriber()

    @property
    def dofs_idx(self):
        if self.actuator_manager:
            return self.actuator_manager.dofs_idx
        return []

    @property
    def joint_names(self):
        if self.actuator_manager:
            return self.actuator_manager.joint_names
        return []
    
    @property
    def _dofs(self):
        if self.actuator_manager:
            return self.actuator_manager._dofs
        return {}

    @property
    def _control_type_cfg(self):
        if self.actuator_manager:
            return self.actuator_manager._control_type_cfg
        return {}

    def _current_sim_timestep(self):
        """
        Get the current sim time
        """
        return self._ros_clock.now().to_msg()

    def _setup_joint_action_publisher(self):
        print("Joint actions Publisher started")
        def joint_action_callback():
                joint_state_msg=JointState()
                joint_state_msg.header.stamp=self._current_sim_timestep()
                joint_state_msg.name=self.joint_names
                # Convert torch tensors to list/numpy for ROS message if needed
                joint_state_msg.position=self._pos_actions.tolist() if isinstance(self._pos_actions, torch.Tensor) else self._pos_actions
                joint_state_msg.velocity=self._vel_actions.tolist() if isinstance(self._vel_actions, torch.Tensor) else self._vel_actions
                joint_state_msg.effort=self._force_actions.tolist() if isinstance(self._force_actions, torch.Tensor) else self._force_actions
                self.joint_state_publisher.publish(joint_state_msg)
                
        self.joint_state_publisher = self._ros_node.create_publisher(JointState, f'/joint_commands', 50)
        self.timer = self._ros_node.create_timer(0.01, joint_action_callback)

    def _setup_robot_twist_subscriber(self):
        print("Robot twist Subscriber started")
        def robot_twist_callback(msg):
            self._robot_lin_vel=msg.linear
            self._robot_ang_vel=msg.angular
        self.robot_twist_subscriber = self._ros_node.create_subscription(Twist, f'/robot_twist', robot_twist_callback, 100)

    def _setup_robot_pose_subscriber(self):
        print("Robot pose Subscriber started")
        def robot_pose_callback(msg):
            self._robot_pos=msg.position
            self._robot_quat=msg.orientation
        self.robot_pose_subscriber = self._ros_node.create_subscription(Pose, f'/robot_pose', robot_pose_callback, 100)

    def _setup_joint_state_subscriber(self):
        print("Joint state Subscriber started")
        def joint_state_callback(msg, joint_properties):
            name_to_pos = {n: p for n, p in zip(msg.name, msg.position)}
            name_to_vel = {n: v for n, v in zip(msg.name, msg.velocity)}
            name_to_eff = {n: e for n, e in zip(msg.name, msg.effort)}
            
            pos_vals = []
            vel_vals = []
            eff_vals = []
            
            # We need to fill values in the order of self.dofs_idx / self.joint_names
            for name in self.joint_names:
                control_type = joint_properties.get(name)
                
                if control_type =="position":
                    pos_vals.append(name_to_pos.get(name, 0.0))
                elif control_type =="velocity":
                    vel_vals.append(name_to_vel.get(name, 0.0))
                elif control_type =='force':
                    eff_vals.append(name_to_eff.get(name, 0.0))
                else:
                    # Default or error
                    pass
            
            self._pos_state=torch.tensor(pos_vals)
            self._vel_state=torch.tensor(vel_vals)
            self._force_state=torch.tensor(eff_vals)

        joint_properties = {}
        for name in self.joint_names:
            control_type = None
            for pattern, value in self._control_type_cfg.items():
                if re.match(f"^{pattern}$", name):
                    control_type = value
                    break
            joint_properties[name] = control_type
            
        self._ros_node.create_subscription(JointState,
                                            f'joint_states',
                                            lambda msg: joint_state_callback(msg, joint_properties), 100)

    # Getters for ObservationManager
    def get_angular_velocity(self):
        # Return tensor of shape (num_envs, 3)
        if self._robot_ang_vel:
            return torch.tensor([self._robot_ang_vel.x, self._robot_ang_vel.y, self._robot_ang_vel.z]).unsqueeze(0)
        return torch.zeros((1, 3))

    def get_linear_velocity(self):
        if self._robot_lin_vel:
            return torch.tensor([self._robot_lin_vel.x, self._robot_lin_vel.y, self._robot_lin_vel.z]).unsqueeze(0)
        return torch.zeros((1, 3))

    def get_projected_gravity(self):
        # Placeholder
        return torch.tensor([0.0, 0.0, -1.0]).unsqueeze(0)

    def get_dofs_position(self):
        if len(self._pos_state) > 0:
            return self._pos_state.unsqueeze(0)
        return torch.zeros((1, len(self.dofs_idx)))

    def get_dofs_velocity(self):
        if len(self._vel_state) > 0:
            return self._vel_state.unsqueeze(0)
        return torch.zeros((1, len(self.dofs_idx)))

def override_config(self,ros_interface):
    self.actuator_manager = ActuatorManager(
        {
        "env":self,
        "joint_names":[
            'FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint'
        ],
        "default_pos":{
            ".*_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        "kp":20,
        "kv":0.5,
        "dofs_limit":{
            ".*_hip_joint": (-1.57, 1.57),
            ".*_thigh_joint": (-1.57, 1.57),
            ".*_calf_joint": (-1.57, 1.57),
        },
        "entity_attr":"robot"
        }
    )
    self.action_manager = PositionActionManager({
        "env":self,
        "scale":0.25,
        "clip":(-100.0, 100.0),
        "use_default_offset":True,
        "actuator_manager":self.actuator_manager,
    })
    self.observation_manager=ObservationManager(
        self,
        cfg={
            "angle_velocity": {
                "fn": lambda env: ros_interface.get_angular_velocity(),
                "scale": 0.25,
            },
            "linear_velocity": {
                "fn": lambda env: ros_interface.get_linear_velocity(),
                "scale": 2.0,
            },
            "projected_gravity": {
                "fn": lambda env: ros_interface.get_projected_gravity(),
            },
            "dof_position": {
                "fn": lambda env: ros_interface.get_dofs_position(),
            },
            "dof_velocity": {
                "fn": lambda env: ros_interface.get_dofs_velocity(),
                "scale": 0.05,
            },
            "actions": {
                "fn": lambda env: env.action_manager.get_actions(),
            },
        },
    )
    self.config_set = True

def main():
    # Processor backend (GPU or CPU)
    rclpy.init()
    
    ros_interface=RosInterface()

    # Load training configuration
    log_path = f"./logs/{args.exp_name}"
    [cfg] = pickle.load(open(f"{log_path}/cfgs.pkl", "rb"))
    model = get_latest_model(log_path)

    def ros_action_handler(self, actions):
        if not self._quiet_action_errors:
            if torch.isnan(actions).any():
                assert NotImplementedError
                print(f"ERROR: NaN actions received! Actions: {actions}")
            if torch.isinf(actions).any():
                print(f"ERROR: Infinite actions received! Actions: {actions}")

        # Process actions
        actions = actions * self._scale_values + self._offset_values
        actions = torch.clamp(
            actions,
            min=self._clip_values[:, 0],
            max=self._clip_values[:, 1],
        )
        
        ros_interface._pos_actions=actions[0, self.actuators.pos_dofs_idx]
        ros_interface._vel_actions=actions[0, self.actuators.vel_dofs_idx]
        ros_interface._force_actions=actions[0, self.actuators.force_dofs_idx]   
        return actions

    # Setup environment
    env = Go2SimpleEnv(num_envs=1, headless=False, env_mode="deploy")
    env.config=types.MethodType(override_config, env)
    env.config(ros_interface)

    # Configure ROS interface early so it has access to managers
    ros_interface.configure(env)

    # Fix: Bind the handler method
    env.action_manager.handle_actions = types.MethodType(ros_action_handler, env.action_manager)


    # print(env.observation_manager)
    env = RslRlWrapper(env)
    env.build()
    
    # Setup ROS topics now that env is built and dofs are ready
    ros_interface.setup_ros()

    # Eval
    print("🎬 Loading last model...")
    runner = OnPolicyRunner(env, cfg, log_path, device=env.device)
    runner.load(model)
    policy = runner.get_inference_policy(device=env.device)

    try:
        obs, _ = env.reset()
        with torch.no_grad():
            while True:
                # Spin ROS node to process callbacks
                time.sleep(0.01)
                rclpy.spin_once(ros_interface, timeout_sec=0)
                # print("obs_mean:",obs)
                actions = policy(obs)
                print("act_mean:",actions)
                obs, _rews, _dones, _infos = env.step(actions)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        raise e
    finally:
        ros_interface.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
