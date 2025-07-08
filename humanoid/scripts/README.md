# script Instruction
## 1.Train
The train.py file serves as the entry point for training. The make_env() function retrieves the robot, environment, and training parameters from the registry based on the user-provided task(e.x. --task=humanoid_ppo). It then obtains the training algorithm and parameters through make_alg_runner(), ultimately calling the training algorithm to perform the training.
#### task registry
The task registration is located in the file `envs/init.py`. It initializes a task with the following parameters: {name: humanoid_ppo, task_class: XBotFreeEnv, env_cfg: XBotCfg, train_cfg: XBotCfgPPO}.
#### sim initialization
`make_env` initializes the simulation environment.The env here has been parallelized, and the return values of the subsequent env methods are combinations of the parallel return values.
#### runner initialization
The initialization of the runner is located in the `make_alg_runner` function. It retrieves the corresponding class name of the runner using `train_cfg_dict["runner_class_name"]` and initializes it. For example, in the `humanoid_ppo` task, the train configuration is `XBotLCfgPPO` (humanoid/envs/custom/humanoid_config), where runner_class_name is defined as "OnPolicyRunner".
#### logs location
By default, the training information and parameters of the model can be found in the `humanoid-gym/logs` folder. The folder name for the logs of a particular run is formatted as `experiment_name/<date_time>_<run_name>`.  For example:
```
python humanoid/scripts/train.py --task=humanoid_ppo --load_run log_file_path --name v1
```
The log file for this run is located at `logs/XBot_ppo/<date_time>_<run_name>_v1`
rained policies are saved in `logs/XBot_ppo/<date_time>_<run_name>/model_<iteration>.pt`.


## 2.play
The play module is used to control the robot in a given environment for task testing with the trained policy. It stores the testing results as an `.mp4` file and saves the speed, position, and actions as `.npz` files. Additionally, it generates charts for the relevant data.
The play module by default uses `ppo_runner.get_inference_policy` to obtain the decision-making model, which is a neural network formed by connecting the normalizer network and the actor network of the PPO. The output of this neural network is the selected action.

#### parameter
- `EXPORT_POLICY` is a boolean variable that determines whether to store the policy as a jit-formatted `.pt` file for optimization or deployment in a Python-independent environment. The storage path is `humanoid-gym/logs/XBot_ppo/exported/policies/policy_1.pt`. The saving and naming of the jit file is located in the `helper.export_policy_as_jit()` method, with a default name of `policy_1.pt`.

- `ENV_NUM` is an integer variable that indicates the maximum number of agents that can be run. 

- `RENDER` is a boolean variable that decides whether to save the simulation visuals of the agents' execution as an .mp4 file, with the storage path being `humanoid-gym/videos/XBot_ppo/`. 

- `RECORD_DATA` is a boolean variable that determines whether to save the position, speed, and actions of the agents at each step as .npz files, with the storage path being the same as that of the loaded model's .pt file.

- `FIX_COMMAND`  is a boolean variable that determines whether to keep the command unchanged at all times.

- `stop_state_log` represents the number of steps run.

## 3.sim2sim
The purpose of this document is to use the policy trained in the Isaac Gym simulation environment for simulations in the MuJoCo environment.

- **Please note: Before initiating the sim-to-sim process, ensure that you run `play.py` to export a JIT policy.**
- **Mujoco-based Sim2Sim Deployment**: Utilize Mujoco for executing simulation-to-simulation (sim2sim) deployments with the command below:
  ```
  python scripts/sim2sim.py --load_model /path/to/export/model.pt
  ```
- **Resolve the conflict between `XBot-L.xml` and the mujoco version: remove the `sensornoise`  from the xml file.**

## 4.play_mujoco
This document is used to run the robot in the MuJoCo environment using the trained policy. It allows for changing commands via keyboard input and plotting graphs of position, velocity, angular velocity, and torque over time.
- **Before running this file, you need to execute `humanoid/algo/ppo/actor_critic.py` to obtain the corresponding ONNX model file.**
- The `BASE_PATH`  needs to be modified as follows.
  ```
  BASE_PATH = path.dirname(path.dirname(path.dirname(path.realpath(__file__))))
  ```
- To enable keyboard input control commands, you need to comment out `play_mujoco.play(None)` and uncomment `curses.wrapper(play_mujoco.play)`
- example:
```
python play_mujoco.py your_path/humanoid-gym/logs/XBot_ppo/Jun24_22-44-18_v1/model_3001.onnx
```
#### plot method
  To enable the relevant drawing functions, you need to uncomment the corresponding methods and modify the relevant parameters as needed.
- `draw_dof_pos(self, dims, draw_action=True)` : Draw the position of the corresponding joints.The `draw_action` determines whether to draw the corresponding action on the same chart.
- `draw_dof_vel(self, dims, draw_action=False)` : Draw the velocity of a joint.
- `draw_euler(self)` : Draw the orientation of the robot.
- `draw_ang_vel(self)` : Draw the angular velocity of the robot.
- `draw_torque(self, dims)` : Draw the torque of the corresponding joints.
    

## 5.analysis_mujoco_isaacgym_gap
This document is used to compare the differences in pose, velocity, and torque between Isaac Gym and MuJoCo under the same conditions and using the same strategy. It also utilizes Matplotlib to generate comparison graphs.
- **Before running this file, you need to run the `play.py` file and set `RECORD_DATA = True` to obtain the corresponding nqz file.**
- The `BASE_PATH`  needs to be modified as follows.
  ```
  BASE_PATH = path.dirname(path.dirname(path.dirname(path.realpath(__file__))))
  ```
- example
```
python analysis_mujoco_isaacgym_gap.py ~/robot/humanoid-gym/logs/XBot_ppo/Jun24_22-44-18_v1/model_3001.npz
```
#### plot method
- `play_mujoco.draw_all_qpos()`: Draws the pose of the root and all joints.
- `play_mujoco.draw_all_torque()`: Draws the torque of all joints.
- `play_mujoco.draw_all_qvel()`: Draws the velocity of the root and all joints.
- `play_mujoco.draw_qpos(dim)`: Draws the pose of the corresponding dimension.
- `play_mujoco.draw_qvel(dim)`: Draws the velocity of the corresponding dimension.