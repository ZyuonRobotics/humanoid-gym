# 工具说明

该文件夹包含多个工具模块，用于任务注册和配置、输入参数处理，以及日志文件的存储和检索。

## task_register
### TaskRegistry()
TaskRegistry() 包含两个方法：`make_env` 用于将环境配置加载到任务中，`make_alg_runner` 用于检索算法运行器信息并将训练配置加载到运行器中。
#### make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default")
`make_alg_runner` 函数用于根据注册信息将训练配置加载到算法运行器中。算法运行器的类在训练配置的 "runner_class_name" 属性中指定。在这里创建运行器的实例，并将 train_cfg 传递给该实例。变量 `resume`、`checkout` 和 `load_run` 可用于在中断后从先前训练的模型继续训练。`resume` 变量是布尔类型，指示是否继续训练上一个模型。`load_run` 指定模型的文件名（默认为最后一个），而 `checkout` 表示加载模型的 `checkpoint` 编号（同样默认为最后一个）。

## terrain
该模块用于根据地形参数配置生成相应的地形。

## helpers
该模块主要用于解析参数、获取路径以及提供一些通用工具。
#### export_policy_as_jit(policy, path)
该函数用于在运行 `play.py` 文件时将策略模型保存为 jit 文件。保存策略的文件名在这里设置，默认值为 `policy_1.pt`。
#### get_load_path(root, load_run=-1, checkpoint=-1):
该函数用于提供与特定运行时间和检查点对应的模型文件路径。参数 `load_run` 表示要加载的文件夹索引，`checkpoint` 表示要加载的模型索引，默认是最近一次运行的最后一个模型检查点。