# 脚本说明

## 1. train
train.py 文件作为训练的入口点。`make_env()` 函数根据用户提供的任务（例如 --task=humanoid_ppo）从注册表中获取机器人、环境和训练参数。然后，它通过 `make_alg_runner()` 获取训练算法和参数，最终调用训练算法进行训练。
#### 任务注册
任务注册位于文件 `envs/init.py` 中。它使用以下参数初始化一个任务：{name: humanoid_ppo, task_class: XBotFreeEnv, env_cfg: XBotCfg, train_cfg: XBotCfgPPO}。
#### 模拟初始化
`make_env` 初始化模拟环境。此处的环境已并行化，后续环境方法的返回值是并行返回值的组合。
#### 运行器初始化
运行器的初始化位于 `make_alg_runner` 函数中。它使用 `train_cfg_dict["runner_class_name"]` 获取相应的运行器类名称并进行初始化。例如，在 `humanoid_ppo` 任务中，训练配置为 `XBotLCfgPPO`（humanoid/envs/custom/humanoid_config），其中 runner_class_name 被定义为 "OnPolicyRunner"。
#### 日志位置
默认情况下，模型的训练信息和参数可以在 `humanoid-gym/logs` 文件夹中找到。特定运行的日志文件夹名称格式为 `experiment_name/<date_time>_<run_name>`。例如：
```
python humanoid/scripts/train.py --task=humanoid_ppo --load_run log_file_path --name v1
```
这个运行的日志文件位于 `logs/XBot_ppo/<date_time>_<run_name>_v1`，训练的策略保存在 `logs/XBot_ppo/<date_time>_<run_name>/model_<iteration>.pt` 中。

## 2. play
play 模块用于在给定环境中控制机器人进行任务测试，使用训练好的策略。它将测试结果存储为 `.mp4` 文件，并将速度、位置和动作保存为 `.npz` 文件。此外，它还生成相关数据的图表。
play 模块默认使用 `ppo_runner.get_inference_policy` 获取决策模型，这是一个由PPO的归一化器网络和`actor`网络连接而成的神经网络。该神经网络的输出就是选择的动作。

#### 参数
- `EXPORT_POLICY` 是一个布尔变量，用于决定是否将策略存储为用于优化或在 Python 独立环境中的部署的 jit 格式 `.pt` 文件。存储路径为 `humanoid-gym/logs/XBot_ppo/exported/policies/policy_1.pt`。jit 文件的保存和命名位于 `helper.export_policy_as_jit()` 方法中，默认名称为 `policy_1.pt`。

- `ENV_NUM` 是一个整数变量，指示可运行的最大代理数量。

- `RENDER` 是一个布尔变量，用于决定是否将代理执行的模拟视觉保存为 `.mp4` 文件，存储路径为 `humanoid-gym/videos/XBot_ppo/`。

- `RECORD_DATA` 是一个布尔变量，用于决定是否在每个步骤保存代理的位置、速度和动作为 `.npz` 文件，存储路径与加载模型的 `.pt` 文件相同。

- `FIX_COMMAND` 是一个布尔变量，用于决定是否始终保持命令不变。

- `stop_state_log` 表示运行的步骤数量。

## 3. sim2sim
本文件的目的是使用在 Isaac Gym 模拟环境中训练的策略进行 MuJoCo 环境中的模拟。

- **请注意：在启动 sim-to-sim 过程之前，请确保运行 `play.py` 导出 JIT 策略。**
- **基于 Mujoco 的 Sim2Sim 部署**：利用 Mujoco 执行模拟到模拟（sim2sim）部署，使用以下命令：
  ```
  python scripts/sim2sim.py --load_model /path/to/export/model.pt
  ```
- **解决 `XBot-L.xml` 和 mujoco 版本之间的冲突：从 xml 文件中移除 `sensornoise`。**

## 4. play_mujoco
该文档用于在 MuJoCo 环境中使用训练好的策略运行机器人。它允许通过键盘输入更改命令，并绘制位置、速度、角速度和扭矩随时间变化的图表。
- **在运行此文件之前，需要执行 `humanoid/algo/ppo/actor_critic.py` 以获取相应的 ONNX 模型文件。**
- `BASE_PATH` 需要修改如下：
  ```
  BASE_PATH = path.dirname(path.dirname(path.dirname(path.realpath(__file__))))
  ```
- 要启用键盘输入控制命令，需要注释掉 `play_mujoco.play(None)` 并取消注释 `curses.wrapper(play_mujoco.play)`。
- 示例：
```
python play_mujoco.py your_path/humanoid-gym/logs/XBot_ppo/Jun24_22-44-18_v1/model_3001.onnx
```
#### 绘图方法
要启用相关绘图功能，需要取消注释对应的方法并根据需要修改相关参数。
- `draw_dof_pos(self, dims, draw_action=True)` : 绘制相应关节的位置。`draw_action` 决定是否在同一图表上绘制相应动作。
- `draw_dof_vel(self, dims, draw_action=False)` : 绘制关节的速度。
- `draw_euler(self)` : 绘制机器人的方向。
- `draw_ang_vel(self)` : 绘制机器人的角速度。
- `draw_torque(self, dims)` : 绘制相应关节的扭矩。

## 5.analysis_mujoco_isaacgym_gap
该文档用于比较在相同条件下使用相同策略的 Isaac Gym 和 MuJoCo 之间的姿态、速度和扭矩差异。它还利用 Matplotlib 生成比较图表。
- **在运行此文件之前，需要运行 `play.py` 文件并设置 `RECORD_DATA = True` 以获取相应的 nqz 文件。**
- `BASE_PATH` 需要修改如下：
  ```
  BASE_PATH = path.dirname(path.dirname(path.dirname(path.realpath(__file__))))
  ```
- 示例：
```
python analysis_mujoco_isaacgym_gap.py ~/robot/humanoid-gym/logs/XBot_ppo/Jun24_22-44-18_v1/model_3001.npz
```
#### 绘图方法
- `play_mujoco.draw_all_qpos()`：绘制根部和所有关节的姿态。
- `play_mujoco.draw_all_torque()`：绘制所有关节的扭矩。
- `play_mujoco.draw_all_qvel()`：绘制根部和所有关节的速度。
- `play_mujoco.draw_qpos(dim)`：绘制相应维度的姿态。
- `play_mujoco.draw_qvel(dim)`：绘制相应维度的速度。