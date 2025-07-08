# custom
The custom folder contains configurations and corresponding environments for specific robots, both of which inherit from the legged-robot, effectively serving as an extension of the legged-robot.

## humanoid_env
#### init()
The `XBotLFreeEnv` class defines the method `create_sim()` for creating the simulation environment, but the initialization is located in its grandparent class `BaseTask`. The `create_sim()` method will create the task environment through `create_envs()`, which is defined in its parent class `LeggedRobot()`.
The initialization steps are as follows:
- `XBotLFreeEnv.init()`
  - `LeggedRobot.init()`
    - `LeggedRobot_parse_cfg()`: Define the time scale, the maximum length of each section, the reward scale, and the command range.
    - `BaseTask().init()`
      - `XBotLFreeEnv.create_sim()`:Create a physical simulation environment.Create environmental terrain.
        - `LeggedRobot._create_envs()`:
          1. loads the robot URDF/MJCF asset,
          2. `LeggedRobot._get_env_origins` Set the starting position of the robot.
          3. `LeggedRobot.init_domain_randomization()` Initialize domain randomization parameters.
          4. For each environment
            4.1 creates the environment, 
            4.2 `LeggedRobot._process_rigid_shape_props()` ,`LeggedRobot._process_dof_props` and `LeggedRobot._process_rigid_body_props`: calls DOF and Rigid shape properties callbacks,
            4.3 create actor with these properties and add them to the env
          5. Store indices of different bodies of the robot
    - `LeggedRobot._init_buffers()`: Initialize torch tensors which will contain simulation states and processed quantities
    - `LeggedRobot._prepare_reward_function()`:`Prepares a list of reward functions, which will be called to compute the total reward.
  - `XBotLFreeEnv.reset_idx()`: Initialize the historical observation values of all envs.
    - `LeggedRobot.reset_idx()`: Reset some environments. Logs episode info. 
      - `LeggedRobot._reset_dofs()`: Resets DOF position and velocities of selected environmments
      - `LeggedRobot._reset_root_states()`: Resets ROOT states position and velocities of selected environmments
      - `LeggedRobot._resample_commands()`: Randommly select commands of some environments
      - [Optional] `LeggedRobot._update_terrain_curriculum()`: Implements the game-inspired curriculum.
      - [Optional] `LeggedRobot.update_command_curriculum()`: Implements a curriculum of increasing commands
  - `XBotLFreeEnv.compute_observations()`: Obtain the observation values of the current environment.

#### step(self, actions)
The `step()` method enables the agent to take the given action and perform a series of simulation simulations, updating the relevant environmental parameters.
The method `step(self, actions)` here adopts domain randomization. The `delay` represents the action's delay. In this context, `self.action` refers to the action executed last time, while `action` denotes the action given by the decision. A weighted sum of the two is computed to represent the delay of the action.
The steps for execution are as follows:
- `XBotLFreeEnv.step()`
  - `LeggedRobot.step()`: Control the degrees of freedom of the robot based on the calculated torque, and perform a simulation.
    - `BaseTask().render()`: Set up a visual interface.
    - `LeggedRobot._compute_torques()`: Compute torques from actions.
    - `LeggedRobot.post_physics_step()`: check terminations, compute observations and rewards.
      - `LeggedRobot._post_physics_step_callback()`: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots.
        - `LeggedRobot._resample_commands()`: Randommly select commands of some environments. 
      - `LeggedRobot.check_termination()`Determine whether to terminate by assessing whether the relevant parts have experienced a force exceeding the threshold and whether the maximum episode has been reached.
      - `LeggedRobot.compute_reward()`:Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function()) adds each terms to the episode sums and to the total reward.
      - `XBotLFreeEnv.reset_idx()`: Reset the envs that reach the termination condition.
      - `XBotLFreeEnv.compute_observations()`:  Add the latest obs to `obs_history` and `critic_history`, and update `obs_buf` and `privileged_obs_buf` to the latest status.
        - `LeggedRobot.get_obs()`  obtain the latest state.
          - `XBotLFreeEnv.compute_ref_state():`The method calculates the reference joint positions of the robot based on the current gait phase and dynamically adjusts them according to the support phase of the left and right feet.
            - `XBotLFreeEnv._get_phase(self):`Calculate the current gait phase.


#### get_symm_obs(self, batch_obs)
Used to obtain the mirror state of the current status, facilitating the generation of mirror-symmetric strategies during reinforcement learning based on the mirror state.

## humanoid_config
The document contains specific parameters for humanoid robots, modified based on the legged-robot configuration.

#### Main changes
- in the `asset` section the path for the robot model file .xml has been set , and penalties and termination points have been added, along with the naming of the feet and knees. 

- In the `control` section, the parameters for the PD controller have been added. 

- In the `reward` section, the scale of the reward and the parameters necessary for calculating the reward have been set.
     