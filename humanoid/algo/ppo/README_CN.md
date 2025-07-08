# PPO 算法

## on_policy_runner
OnPolicyRunner 类的目的是在强化学习环境中执行基于策略的学习过程。它主要由两个部分组成：学习和返回策略。

#### learn(self, num_learning_iterations, init_at_random_ep_len=False)
该方法负责学习和更新参数。
由于在环境 (env) 中使用了并行化，`obs`,`privileged_obs`,`rewards`,`dones`,`infos` 和 `actions` 的维度均为 (envs, num_per_env)。例如，当 num_envs = 4096 且 num_obs = 12 时，这里的 `obs` 的维度将为 (4096, 12)。
`learn` 函数的第二个循环完成了一次完整的采样序列。它调用 `self.num_steps_per_env` 步进行采样，将每一步的 `obs`,`privileged_obs`,`rewards`,`dones`,`infos`  记录到 `Transition` 中. 该步的 `Transition` 存储到 `RolloutStorage` 中。完成环境的步数后，它通过使用 `self.alg.compute_returns()` 计算执行过程中的所有动作的优势。
完成一个采样序列后，使用 `self.alg.update` 来更新参数。

#### get_inference_policy(self, device=None)
get_inference_policy 函数用于获取用于推理（评估或预测）的策略网络。
- 1.获取策略网络：从 self.alg.actor_critic 提取 actor（策略网络），用于生成动作。
- 2.经验归一化：如果在配置中启用了经验归一化 (self.cfg["empirical_normalization"])，将会将归一化层与策略网络结合形成一个新的顺序模型。这意味着在推理过程中，原始观测值将在输入到策略网络之前先进行归一化。

最终返回结果是由观测归一化网络和 actor 网络组成的神经网络。

#### save(self, path, infos=None)
save 函数将训练好的模型转换为包含以下状态的字典，并将其保存在 .pt 文件中。
`model_state_dict`: actor、critic 网络参数,
`optimizer_state_dict`: 优化器参数,
`iter`: current_learning_iteration,
`infos`: infos,
`obs_norm_state_dict` = obs_normalizer,
`privileged_obs_norm_state_dict` = privileged_obs_normalizer

## ppo
本模块是 PPO 算法的核心部分。
#### act(self, obs, critic_obs) 和 process_env_step(self, rewards, dones, infos)
`act` 方法调用 ActorCritic 模块来采样一个动作，将状态和动作信息存储在 `Transition` 中，并返回选择的动作。
`process_env_step` 方法用于在代理执行动作后，将环境中的奖励、终止状态和信息存储在 `Transition` 中。然后将转移保存到 `RolloutStorage` 中。
`act` 和 `process_env_step` 方法共同完成一次回合的采样和存储。
#### update(self)
`update()` 方法涉及计算目标函数来更新 actor 网络和 critic 网络。
- `KL` : KL 散度用于控制优化器的学习率，调节更新幅度，以防止新策略过度偏离旧策略，从而导致训练不稳定。
  `kl = torch.sum(torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)`
$$KL(\pi_{old}||\pi)=\mathbb{E}_{x\sim N(\mu_{old},\sigma_{old})}[\log \frac{\pi_{old}}{\pi}]=\mathbb{E}_{x\sim N(\mu_{old},\sigma_{old})}[\log \frac{\sigma}{\sigma_{old}}-\frac{(x-\mu_{old})^{2}}{2\sigma_{old}^{2}}+\frac{(x-\mu)^{2}}{2\sigma^{2}}]=\log \frac{\sigma}{\sigma_{old}}-\frac{\mathbb{E}[(x-\mu_{old})^{2}]}{2\sigma_{old}^{2}}+\frac{\mathbb{E}(x-\mu)^{2}}{2\sigma^{2}}$$
$$=\log \frac{\sigma}{\sigma_{old}}-0.5+\frac{\mathbb{E}(x^{2})-2\mu\mathbb{E}(x)+\mu^{2}}{2\sigma^{2}}=\log \frac{\sigma}{\sigma_{old}}-0.5+\frac{\sigma_{old}^{2}+\mu_{old}^{2}-2\mu\mu_{old}+\mu^{2}}{2\sigma^{2}}=\log \frac{\sigma}{\sigma_{old}}-0.5+\frac{\sigma_{old}^{2}+(\mu-\mu_{old})^{2}}{2\sigma^{2}}$$
- `surrogate_loss` :损失函数计算从新动作中获得的期望优势。
- `Value_loss` : 损失是状态值函数的损失，用于更新 critic 网络。
- `symm_loss`: 损失表示代理在镜像状态下采取的动作与原始动作的镜像动作之间的差异。这个损失用于指导策略网络生成对称策略，同时加速优化。

## actor_critic
ActorCritic 表示强化学习中的 Actor-Critic 框架。它由两个神经网络组成：Actor 和 Critic。
#### attribute
`self.actor` 和 `self.critic`
Actor 网络的输入是代理观察到的参数，记作 obs，输出是每个维度的动作（例如，在 XBotLCfg 中，num_active_dofs = 12，总共有 12 维的动作）。Critic 网络的输入是从上帝视角观察到的参数，记作 critic_obs，输出是表示该状态的奖励的值。

`self.distribution`
`self.distribution` 是动作的正态分布。在训练过程中，动作是从这个分布中采样的。该分布是基于 Actor 返回的结果作为均值构造的，使用自定义的 init_noise_std 作为方差。
#### act(self, observations, **kwargs)
在训练过程中，"act" 被用来选择动作。由于代理的动作是连续的，因此需要使用均值和方差构造正态分布作为动作的概率密度函数。首先将观察值 (obs) 输入自我演员神经网络，使用结果作为正态分布的均值。然后，建立一个多维正态分布，最后从该分布中采样以获得最终动作。
#### get_actions_log_prob(self, actions)
由于在训练过程中使用正态分布采样选择动作，因此需要估算选择给定动作的概率。此方法用于估算该动作的概率。

## rollout_storage
`RolloutStorage` 类用于存储通过环境交互生成的数据，执行优势函数的反向计算，并生成批量数据。
#### Transition 
`Transition` 类记录一次交互的数据。critic_observations 和 observations 都是采取动作之前的状态。
#### mini_batch_generator(self, num_mini_batches, num_epochs=8)
`mini_batch_generator` 的返回值是一个迭代器，每次都可以从中获取一批样本。由于 PPO 优化过程不受步骤顺序的影响，因此在样本处理期间，`RolloutStorage` 中的一些属性通过 flatten(0, 1) 操作被展平。该操作将步骤维度和环境维度合并为一个维度。例如，self.action 的原始维度是 (num_steps, num_envs, 12)，展平后变为 (num_steps * num_envs, 12)。
#### compute_returns(self, last_values, gamma, lam)
`compute_returns` 方法通过反向计算计算每个样本状态和动作的 Q 值和优势，并对优势进行归一化。
$$\delta = R_{t}+\gamma V(s_{t+1})-V(s_{t})\qquad A(s_{t},a_{t})=\delta+\gamma*\lambda*A(s_{t+1},a_{t+1})\qquad Q(s_{t},a_{t})=A(s_{t},a_{t})+V(s_{t})$$

## normalizer
该模块用于对输入数据进行归一化，从而加快收敛并提高训练期间的稳定性。
#### EmpiricalNormalization(nn.Module):
`EmpiricalNormalization` 用于在训练期间估计每个状态维度的经验均值和经验方差，并对训练样本进行归一化。在推理测试期间，输入数据根据训练期间获得的经验均值和经验方差进行经验归一化。

#### EmpiricalDiscountedVariationNormalization(nn.Module):
`EmpiricalDiscountedVariationNormalization` 用于奖励归一化。由于奖励函数是非平稳的，对奖励的尺度进行归一化是有用的，以便值函数可以快速学习。我们通过将奖励除以折扣奖励总和的标准差的运行估计来实现这一点。