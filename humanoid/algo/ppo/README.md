# PPO alogrithm

## on_policy_runner
The purpose of the OnPolicyRunner class is to execute policy-based learning processes in a reinforcement learning environment. It mainly consists of two parts: learning and return policy.

#### learn(self, num_learning_iterations, init_at_random_ep_len=False)
The method is responsible for the learning and updating of parameters.
Due to the use of parallelization in the environment (env), the dimensions of `obs`, `privileged_obs`, `rewards`, `dones`, `infos`, and `actions` are all (envs, num_per_env). For example, when num_envs = 4096 and num_obs = 12, the dimension of `obs` here would be (4096, 12).
The second loop of the learn function completes one full sequence of sampling. It performs sampling for self.num_steps_per_env steps, recording each step's obs, privileged_obs, rewards, dones, and infos into the `Transition`, and then storing the `Transition` of that step into `RolloutStorage`. After the steps for that environment are completed, it calculates the advantages for all actions taken during the steps by using `self.alg.compute_returns()`.
After a sequence of sampling is complete, use `self.alg.update` to update the parameters.

#### get_inference_policy(self, device=None)
The function get_inference_policy serves to obtain the policy network used for inference (evaluation or prediction).
- 1.Acquiring the policy network: The actor (policy network) is extracted from self.alg.actor_critic, which is used to generate actions.
- 2.Experience normalization: If empirical normalization is enabled in the configuration (self.cfg["empirical_normalization"]), a normalization layer will be combined with the policy network to form a new sequential model. This means that during inference, the original observations will first be normalized before being fed into the policy network.

The final return value is a neural network composed of the observation normalization network and the actor network.

#### save(self, path, infos=None)
The save function converts the trained model into a dictionary with the following state and saves it in a .pt file.
    `model_state_dict`: actor,critic network parameters ,
    `optimizer_state_dict`: optimizer parameters,
    `iter`: current_learning_iteration,
    `infos`: infos,
    `obs_norm_state_dict` = obs_normalizer,
    `privileged_obs_norm_state_dict` = privileged_obs_normalizer

## ppo
This module is the main part of the PPO algorithm.
#### act(self, obs, critic_obs) and process_env_step(self, rewards, dones, infos)
The `act` method calls the ActorCritic module to sample an action, stores the state and action information in the `Transition`, and returns the selected action.
The `process_env_step` method is used to store the reward, terminal state, and information from the environment in the `Transition` after the agent takes the action given by `act` . It then saves the transition to `RolloutStorage`. 
The `act` and `process_env_step` methods together complete the sampling and storage of one rollout.
#### update(self)
`update()` involves calculating the objective function to update the actor network and the critic network.
- `KL` : The KL divergence is used to control the learning rate of the optimizer, regulating the update magnitude to prevent the new policy from deviating too much from the old policy, which could lead to training instability.
  `kl = torch.sum(torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)`
$$KL(\pi_{old}||\pi)=\mathbb{E}_{x\sim N(\mu_{old},\sigma_{old})}[\log \frac{\pi_{old}}{\pi}]=\mathbb{E}_{x\sim N(\mu_{old},\sigma_{old})}[\log \frac{\sigma}{\sigma_{old}}-\frac{(x-\mu_{old})^{2}}{2\sigma_{old}^{2}}+\frac{(x-\mu)^{2}}{2\sigma^{2}}]=\log \frac{\sigma}{\sigma_{old}}-\frac{\mathbb{E}[(x-\mu_{old})^{2}]}{2\sigma_{old}^{2}}+\frac{\mathbb{E}(x-\mu)^{2}}{2\sigma^{2}}$$
$$=\log \frac{\sigma}{\sigma_{old}}-0.5+\frac{\mathbb{E}(x^{2})-2\mu\mathbb{E}(x)+\mu^{2}}{2\sigma^{2}}=\log \frac{\sigma}{\sigma_{old}}-0.5+\frac{\sigma_{old}^{2}+\mu_{old}^{2}-2\mu\mu_{old}+\mu^{2}}{2\sigma^{2}}=\log \frac{\sigma}{\sigma_{old}}-0.5+\frac{\sigma_{old}^{2}+(\mu-\mu_{old})^{2}}{2\sigma^{2}}$$
- `surrogate_loss` :The loss function calculates the expected advantage gained from the new action.
- `Value_loss` : The loss is the loss of the state value function, used to update the critic network.
- `symm_loss`: The loss represents the difference between the actions taken by the agent in the mirror state and the mirrored actions of the original actions. This loss is used to guide the policy network to generate symmetric policies, while also accelerating optimization.


## actor_critic
ActorCritic represents the Actor-Critic framework in reinforcement learning. It consists of two neural networks: the Actor and the Critic. 
#### attribute
`self.actor` and `self.critic` 
The input to the Actor network is the parameters observed by the agent, denoted as obs, and the output is the actions for each dimension (for example, in XBotLCfg, with num_active_dofs = 12, there are 12 dimensions of actions in total). The input to the Critic network is the parameters observed from a god-like perspective, denoted as critic_obs, and the output is a value that represents the reward for that state.

`self.distribution`
`self.distribution` is a normal distribution of actions. During training, actions are sampled from this distribution.The distribution is constructed based on the results returned by the Actor as the mean, with a custom init_noise_std as the variance.
#### act(self, observations, **kwargs)
During the training process, "act" is used to select actions. Due to the continuous nature of the agent's actions, it is necessary to construct a normal distribution as the probability density function of the actions using the mean and variance. First input the observation (obs) into the self.actor neural network, using the result as the mean of a normal distribution. Then, establish a multidimensional normal distribution and finally sample from this distribution to obtain the final action.
#### get_actions_log_prob(self, actions)
Due to the use of normal distribution sampling to select actions during training, it is necessary to estimate the probability of selecting a given action. This method is used to estimate the probability of that action.

## rollout_storage
The `RolloutStorage` class is used to store data generated by environment interactions, perform backward calculation of advantage functions, and generate batch data.
#### Transition 
The `Transition` class records the data of an interaction. critic_observations and observations are both the states before taking action
#### mini_batch_generator(self, num_mini_batches, num_epochs=8)
The return value of the `mini_batch_generator` is an iterator, from which a batch of samples can be obtained each time. Since the PPO optimization process is not affected by the order of steps, some attributes in `RolloutStorage` are flattened using the flatten(0, 1) operation during the sample processing. This operation combines the step dimension and the envs dimension into one dimension. For example, the original dimension of self.action is (num_steps, num_envs, 12), and after flattening, it becomes (num_steps * num_envs, 12).
#### compute_returns(self, last_values, gamma, lam)
The `compute_returns` method calculates the Q-values and advantages for each sample state and action through backward computation, and normalizes the advantages.
$$\delta = R_{t}+\gamma V(s_{t+1})-V(s_{t})\qquad A(s_{t},a_{t})=\delta+\gamma*\lambda*A(s_{t+1},a_{t+1})\qquad Q(s_{t},a_{t})=A(s_{t},a_{t})+V(s_{t})$$

## normalizer
This module is used for normalizing input data, which accelerates convergence and increases stability during training.
#### EmpiricalNormalization(nn.Module):
`EmpiricalNormalization `is used to estimate the empirical mean and empirical variance for each dimension of the state during training, and to normalize the training samples. During inference testing, the input data is normalized empirically based on the empirical mean and empirical variance obtained during training.

#### EmpiricalDiscountedVariationNormalization(nn.Module):
   EmpiricalDiscountedVariationNormalization is used to Reward normalization. Since the reward function is non-stationary, it is useful to normalize the scale of the rewards so that the value function can learn quickly. We did this by dividing the rewards by a running estimate of the standard deviation of the sum of discounted rewards.
    


