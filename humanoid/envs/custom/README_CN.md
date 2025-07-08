# custom

custom 文件夹包含特定机器人所需的配置和相应环境，这些环境均继承自`legged-robot`，实际上作为`legged-robot`的扩展。

## humanoid_env
#### init()
`XBotLFreeEnv` 类定义了 `create_sim()` 方法，用于创建模拟环境，但初始化位于其祖父类 `BaseTask` 中。`create_sim()` 方法通过 `create_envs()` 来创建任务环境，该方法在其父类 `LeggedRobot()` 中定义。
初始化步骤如下：
- `XBotLFreeEnv.init()`
  - `LeggedRobot.init()`
    - `LeggedRobot_parse_cfg()`：定义时间尺度、每段的最大长度、奖励尺度和指令范围。
    - `BaseTask().init()`
      - `XBotLFreeEnv.create_sim()`: 创建物理模拟环境。创建环境地形。
        - `LeggedRobot._create_envs()`:
          1. 加载机器人 URDF/MJCF 资产，
          2. `LeggedRobot._get_env_origins` 设置机器人的起始位置。
          3. `LeggedRobot.init_domain_randomization()` 初始化领域随机化参数。
          4. 对每个环境
            4.1 创建环境，
            4.2 `LeggedRobot._process_rigid_shape_props()`、`LeggedRobot._process_dof_props` 和 `LeggedRobot._process_rigid_body_props`：调用 DOF 和刚体形状属性的回调，
            4.3 根据这些属性创建演员并将其添加到环境中。
          5. 存储机器人不同部位的索引。
    - `LeggedRobot._init_buffers()`：初始化将包含仿真状态和处理数量的 Torch 张量。
    - `LeggedRobot._prepare_reward_function()`：准备奖励函数列表，将在计算总奖励时调用。
  - `XBotLFreeEnv.reset_idx()`：初始化所有环境的历史观测值。
    - `LeggedRobot.reset_idx()`：重置某些环境。记录情节信息。
      - `LeggedRobot._reset_dofs()`：重置所选环境的 DOF 位置和速度。
      - `LeggedRobot._reset_root_states()`：重置所选环境的根状态位置和速度。
      - `LeggedRobot._resample_commands()`：随机选择指令。
      - [可选] `LeggedRobot._update_terrain_curriculum()`：逐步增加地形难度。
      - [可选] `LeggedRobot.update_command_curriculum()`：逐步增加指令难度。
  - `XBotLFreeEnv.compute_observations()`：获取当前环境的观测值。

#### step(self, actions)
`step()` 方法使代理能够执行给定的动作并进行一系列模拟仿真，更新相关的环境参数。
这里的 `step(self, actions)` 方法采用领域随机化。`delay` 表示该动作的延迟。在这种情况下，`self.action` 指的是上一次执行的动作，而 `action` 表示决策给定的动作。计算这两者的加权和以表示动作的延迟。
执行步骤如下：
- `XBotLFreeEnv.step()`
  - `LeggedRobot.step()`：根据计算出的扭矩控制机器人的自由度，并进行模拟。
    - `BaseTask().render()`：设置视觉接口。
    - `LeggedRobot._compute_torques()`：根据动作计算扭矩。
    - `LeggedRobot.post_physics_step()`：检查终止条件，计算观测值和奖励。
      - `LeggedRobot._post_physics_step_callback()`：根据目标和朝向计算角速度命令，计算测量的地形高度并随机推机器人。
        - `LeggedRobot._resample_commands()`：随机选择某些环境的指令。
      - `LeggedRobot.check_termination()`：通过评估相关部分是否经历超过阈值的力以及是否达到最大情节来确定是否终止。
      - `LeggedRobot.compute_reward()`：调用每个具有非零尺度的奖励函数（在 self._prepare_reward_function() 中处理），将每个项加到情节总和和总奖励中。
      - `XBotLFreeEnv.reset_idx()`：重置到达终止条件的环境。
      - `XBotLFreeEnv.compute_observations()`：将最新的观测值添加到 `obs_history` 和 `critic_history` 中，并更新 `obs_buf` 和 `privileged_obs_buf` 到最新状态。
        - `LeggedRobot.get_obs()` 获取最新状态。
          - `XBotLFreeEnv.compute_ref_state()`：该方法根据当前步态阶段计算机器人的参考关节位置，并根据左右脚的支撑阶段动态调整它们。
            - `XBotLFreeEnv._get_phase(self)`：计算当前步态阶段。

#### get_symm_obs(self, batch_obs)
用于获取当前状态的镜像状态，便于在强化学习中基于镜像状态生成镜像对称策略。

## humanoid_config
该文档包含针对人形机器人的特定参数，基于腿部机器人的配置进行了修改。

#### 主要更改
- 在 `asset` 部分设置了机器人模型文件 .xml 的路径，并添加了惩罚和终止点，同时命名了脚和膝盖。

- 在 `control` 部分，添加了 PD 控制器的参数。

- 在 `reward` 部分，设置了奖励的尺度和计算奖励所需的参数。