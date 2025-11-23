import os
import copy
import torch
import shutil
import pickle
import argparse
from importlib import metadata
import genesis as gs

from genesis_forge.wrappers import (
    VideoWrapper,
    SkrlEnvWapper,
)
from environment import Go2SimpleEnv

import torch.nn as nn

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.utils.runner.torch import Runner
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer


import torch
import torch.nn as nn

class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)
        # print("obs space:",self.observation_space)
        # exit(0)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.net = nn.Sequential(
            nn.Linear(109, 64),
            nn.ELU()
        )

        self.mean_layer = nn.Linear(64, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions))
        self.value_layer = nn.Linear(64, 1)
        self._shared_output = None

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "policy":
            states = inputs["states"]
            # print("obs space:",self.observation_space)
            # print("states:",states)
            # exit(0)
            space = self.tensor_to_space(states, self.observation_space)
            # print("front_image_shape",space.shape)
            image_features = self.conv_layers(space["front_image"].permute(0, 3, 1, 2).float())
            # Apply global average pooling across spatial dimensions (height and width)
            image_features = image_features.mean(dim=[2, 3])  # Global average pooling (across height and width)
            # Flatten the tensor to have a single dimension of features
            image_features = torch.flatten(image_features, 1)
            proprio_features = torch.cat([
                space[key].view(states.shape[0], -1) for key in space.keys() if key != "front_image"
            ], dim=1)

            self._shared_output = self.net(torch.cat([image_features, proprio_features], dim=1))

            return self.mean_layer(self._shared_output), self.log_std_parameter, {}

        elif role == "value":
            if self._shared_output is None:
                states = inputs["states"]
                space = self.tensor_to_space(states, self.observation_space)
                image_features = self.conv_layers(space["front_image"].permute(0, 3, 1, 2).float())
                # Apply global average pooling across spatial dimensions (height and width)
                image_features = image_features.mean(dim=[2, 3])  # Global average pooling (across height and width)
                # Flatten the tensor to have a single dimension of features
                image_features = torch.flatten(image_features, 1)
                proprio_features = torch.cat([
                    space[key].view(states.shape[0], -1) for key in space.keys() if key != "front_image"
                ], dim=1)

                self._shared_output = self.net(torch.cat([image_features, proprio_features], dim=1))

            return self.value_layer(self._shared_output), {}

        
EXPERIMENT_NAME = "go2-camera"

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("-n", "--num_envs", type=int, default=25)
parser.add_argument("--max_iterations", type=int, default=1000)
parser.add_argument("-d", "--device", type=str, default="gpu")
parser.add_argument("-e", "--exp_name", type=str, default=EXPERIMENT_NAME)
args = parser.parse_args()


def main():
    # Initialize Genesis
    # Processor backend (GPU or CPU)
    backend = gs.gpu
    if args.device == "cpu":
        backend = gs.cpu
        torch.set_default_device("cpu")
    gs.init(logging_level="warning", backend=backend)

    # Logging directory
    log_base_dir = "./logs"
    experiment_name = args.exp_name
    log_path = os.path.join(log_base_dir, experiment_name)
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path, exist_ok=True)
    print(f"Logging to: {log_path}")

    # Create environment
    env = Go2SimpleEnv(num_envs=args.num_envs, headless=True,deploy_with_ros=False)

    # Record videos in regular intervals
    env = VideoWrapper(
        env,
        video_length_sec=12,
        out_dir=os.path.join(log_path, "videos"),
        episode_trigger=lambda episode_id: episode_id % 5 == 0,
    )

    # Build the environment
    env = SkrlEnvWapper(env)
    env.build()
    env.reset()
    
    memory = RandomMemory(memory_size=4, num_envs=env.num_envs, device=gs.device)
    
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 4  # memory_size
    cfg["learning_epochs"] = 5
    cfg["mini_batches"] = 2  # 96 * 4096 / 98304
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 1e-3
    cfg["learning_rate_scheduler"] = KLAdaptiveLR
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01, "min_lr": 5e-4}
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0
    cfg["grad_norm_clip"] = 1.0
    cfg["ratio_clip"] = 0.2
    cfg["value_clip"] = 0.2
    cfg["clip_predicted_values"] = True
    cfg["entropy_loss_scale"] = 0.01
    cfg["value_loss_scale"] = 1.0
    cfg["kl_threshold"] = 0
    cfg["rewards_shaper"] = None
    cfg["time_limit_bootstrap"] = True
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": gs.device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": gs.device}
    # logging to TensorBoard and write checkpoints (in timesteps)
    cfg["experiment"]["write_interval"] = 60
    cfg["experiment"]["checkpoint_interval"] = 100
    models = {}
    models["policy"] = Shared(env.observation_space, env.action_space, gs.device)
    models["value"] = models["policy"]  # same instance: shared model
    agent = PPO(models=models,
                memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=gs.device)
    cfg_trainer = {"timesteps": args.max_iterations*cfg["learning_epochs"], "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    # Train
    print("💪 Training model...")
    trainer.train()
    env.close()


if __name__ == "__main__":
    main()
