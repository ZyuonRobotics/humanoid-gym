# base

此文件夹包含基本环境、配备的机器人配置以及创建相应任务的模拟环境，并控制机器人与环境的交互。

## base_config
本类仅有一个方法 `init_member_classes(obj)`，该方法能够在类初始化过程中实例化所有子类的内部类。例如，`XBotLCfg` 是 `BaseConfig` 的一个子类，`Terrain` 是 `XBotLCfg` 的一个内部类。当 `BaseConfig` 被初始化时，将创建一个 `terrain` 类的实例。

## base_task
BaseTask 包括创建模拟环境的基本框架，并在指定任务下与其交互。在此处，初始化了并行实例的数量、状态数量、动作数量以及相应的缓冲区。BaseTask 还初始化了模拟环境，其中 `self.create_sim()` 指的是子类的 `create_sim()` 方法。
在初始化 BaseTask 的过程中，它还会根据输入参数 `headless` 确定是否创建用于观察模拟环境的视图。
#### render(self, sync_frame_time=True)
`render` 方法用于渲染图形界面和处理用户输入事件。在执行过程中，代码检查窗口状态，处理用户的键盘事件，并根据模拟结果更新图形内容。

## legged_robot_config
该文档包含模拟环境、机器人、控制器和训练器的完整腿部机器人配置。
#### LeggedRobotCfg()
`LeggedRobotCfg()` 类包含与模拟环境、腿部机器人和控制器相关的所有配置。
- `env` 包含状态信息、动作信息、并行数量、每秒的情节数量以及获取当前状态的方法。
- `terrain` 包含模拟地形的配置。
- `commands` 包含给予代理的指令或目标，以指导其行为。这些指令由四种类型组成：lin_vel_x、lin_vel_y、ang_vel_yaw 和 heading，而 `range` 内部类约束了这些指令的限制。
- `init_state` 定义代理的初始位置、速度和旋转角度。
- `control` 定义控制器的相关参数。
- `asset` 定义腿部机器人的相关物理属性，包括碰撞属性、阻尼属性、重力属性、速度限制和固有特性。
- `domain_rand` 定义相关物理属性的随机化范围，领域随机化可以帮助模型更好地从模拟转移到现实（sim2real）。
- `rewards` 定义与强化学习奖励相关的配置，其中尺度指定每个状态奖励的系数。
- `Normalization` 定义观测状态的阈值。
- `viewer` 定义视图中相机的位置和方向。
- `sim` 定义与模拟和物理引擎设置相关的配置。
#### LeggedRobotCfgPPO(BaseConfig)
`LeggedRobotCfgPPO` 定义与 PPO 算法训练相关的配置。

## legged_robot
该文档包含控制代理的逻辑、处理环境状态、与环境交互以及处理环境反馈的逻辑。
**总体操作流程可以在 custom/README.md 中找到。**