# utils Instruction
The folder contains several tool modules for task registration and configuration, input parameter processing, and storage and retrieval of log files.

## task_register
### TaskRegistry()
 TaskRegistry() contains two methods: `make_env` for loading the environment configuration into the task, and `make_alg_runner` for retrieving the algorithm runner information and loading the training configuration into the runner.
#### make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default")
The function `make_alg_runner `is used to load the training configuration into the algorithm runner based on the registry information. The class of the algorithm runner is specified in the "runner_class_name" attribute of the training configuration. An instance of the runner is created here, and the train_cfg is passed into the instance. The variables `resume`, `checkout`, and `load_run` can be used to continue training from a previously trained model after an interruption. The `resume` variable is of boolean type, indicating whether to continue the training of the last model. `load_run` specifies the filename of the model (defaulting to the last one), while checkout denotes the `checkpoint` number of the loaded model (also defaulting to the last one).

## terrain
This module is used to generate corresponding terrain based on terrain parameter configuration.

## helpers
The module is mainly used for parsing parameters, obtaining paths, and providing some general-purpose tools.
#### export_policy_as_jit(policy, path)
This function is used to save the policy model as a jit file when running the `play.py` file. The filename for the saved policy is set here, with a default value of `policy_1.pt`.
#### get_load_path(root, load_run=-1, checkpoint=-1):
This function is used to provide the path of the model file corresponding to a specific run time and checkpoint. The parameter `load_run` represents the index of the folder to load, and `checkpoint` represents the index of the model to load, with the default being the last model checkpoint from the most recent run.



