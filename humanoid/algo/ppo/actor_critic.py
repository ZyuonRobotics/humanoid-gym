# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        init_noise_std=1.0,
                        activation = nn.ELU(),
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()


        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

if __name__ == '__main__':
    from argparse import ArgumentParser
    from normalizer import EmpiricalNormalization
    import onnxruntime
    import numpy as np

    parser = ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--single_obs_num", default=47, type=int)
    parser.add_argument("--single_privileged_obs_num", default=74+4+60, type=int)
    parser.add_argument("--action_num", default=12, type=int)
    parser.add_argument("--frame_stack", default=15, type=int)
    parser.add_argument("--c_frame_stack", default=3, type=int)

    args = parser.parse_args()
    state_dict = torch.load(args.path, weights_only=True)
    normalized = "obs_norm_state_dict" in state_dict

    actor_critic = ActorCritic(
        args.single_obs_num * args.frame_stack,
        args.single_privileged_obs_num * args.c_frame_stack,
        args.action_num,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[768, 256, 128],
    )
    actor_critic.load_state_dict(state_dict["model_state_dict"])
    actor_critic.eval()

    if normalized:
        obs_normalizer = EmpiricalNormalization(shape=(args.single_obs_num * args.frame_stack), until=int(1.0e8))
        obs_normalizer.load_state_dict(state_dict["obs_norm_state_dict"])
        print("normalizer mean:", obs_normalizer.mean[-args.single_obs_num:])
        print("normalizer std:", obs_normalizer.std[-args.single_obs_num:])
        obs_normalizer.eval()
        policy = torch.nn.Sequential(obs_normalizer, actor_critic.actor)
    else:
        obs_normalizer = None
        policy = actor_critic.actor

    inputs = torch.ones([1, args.single_obs_num * args.frame_stack])
    if normalized:
        normalized_inputs = obs_normalizer(inputs)
    else:
        normalized_inputs = inputs
    actions = actor_critic.actor(normalized_inputs)

    print("inputs:", inputs[0, -args.single_obs_num:])
    print("normalized_inputs:", normalized_inputs[0, -args.single_obs_num:])
    print("actions:", actions[0])

    onnx_path = args.path.replace('.pt', '.onnx')
    torch.onnx.export(policy, inputs, onnx_path)

    ort_session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = ort_session.get_inputs()[0].name
    actions_np = ort_session.run(None, {
        input_name: np.ones([1, args.single_obs_num * args.frame_stack], dtype=np.float32)
    })[0]
    print("actions_np:", actions_np[0])

    assert np.allclose(actions_np, actions.detach().numpy(), atol=0.001)

