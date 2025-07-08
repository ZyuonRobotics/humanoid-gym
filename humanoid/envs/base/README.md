# Base
This folder contains the basic environment, the equipped robot configuration, and the simulation environment for creating corresponding tasks, as well as controlling the robot's interaction with the environment.

## base_config
This class has only one method, `init_member_classes(obj)`, which is capable of instantiating all subclass's inner Class during the initialization of the class. For example, `XBotLCfg` is a subclass of `BaseConfig`. `Terrain` is an inner class of `XBotLCfg`. When `BaseConfig` is initialized, an instance of the `terrain` class will be created.

## base_task
BaseTask includes the basic framework for creating a simulation environment and interacting with it under specified tasks. Here, the number of parallel instances, the number of states, the number of actions, and the corresponding buffers are initialized. BaseTask also initializes the simulation environment, where `self.create_sim()` refers to the `create_sim()` method of the subclass.
During the initialization of BaseTask, it also determines whether to create a view for observing the simulation environment based on the input parameter `headless`.
#### render(self, sync_frame_time=True)
The `render` method is used to render the graphical interface and handle user input events. During execution, the code checks the window state, processes user keyboard events, and updates the graphical content based on the simulation results.

## legged_robot_config
The document contains the complete legged robot configurations for the simulation environment, robot, controller, and trainer.
#### LeggedRobotCfg()
The `LeggedRobotCfg()` class contains all the configurations related to the simulation environment, the legged robot, and the controllers.
- `env` contains state information, action information, parallel quantities, the number of episodes per second in reinforcement learning, as well as a method for obtaining the current state.
- `terrain` includes the configuration of the simulated terrain.
- `commands` includes the instructions or objectives given to the agent in the environment to guide its behavior. The instructions consist of four types: lin_vel_x, lin_vel_y, ang_vel_yaw, and heading, while  `range` inner class constrains the limits of these commands.
- `init_state` defines the agent's initial position, velocity, and rotation angle.
- `control` defines the relevant parameters of the controller.
- `asset` defines the relevant physical properties of the legged robot, including collision properties, damping properties, gravity properties, speed limits, and intrinsic characteristics.
- `domain_rand` defines the range of randomization for related physical properties, and domain randomization can help the model better transfer from simulation to reality (sim2real).
- `rewards` define the configuration related to reinforcement learning rewards, where the scale specifies the coefficient for each state's reward.
- `Normalization` defines the thresholds for the observed states.  
- `viewer` defines the position and direction of the camera in the view.  
- `sim` defines the configuration related to simulation and the physics engine settings.
#### LeggedRobotCfgPPO(BaseConfig)
`LeggedRobotCfgPPO` defines the configuration related to the training of the PPO algorithm.

## legged_robot
The document contains the logic for controlling the agent, processing the environment's states, interacting with the environment, and handling feedback from the environment.
**The overall operation process can be found in the custom/README.md .**

