# humanoid Instruction

## constrction

├── algo                          # RL algorithm
│   ├── __init__.py              
│   ├── vec_env.py               
│   └── ppo                       # PPO algorithm 
│                                  
├── envs                          # robot/environment config
│   ├── __init__.py               # init config and register task
│   ├── base                      # base config / legged robot config (base class) / base environment
│   └── custom                    # humanoid config (child class) / humanoid environment
│                                  
├── scrpit                       
│   ├── train.py                  # RL train
│   ├── play.py                   # Run the trained strategy in the Isaac Gym environment.
│   ├── play_mujoco.py            # Run the trained strategy in the mujoco environment.
│   ├── analysis_mujoco_isaacgym_gap.py  # Calculate the differences between Mujoco and Isaac Gym.
│   └── sim2sim.py                # Run MuJoCo simulation under custom commands.
│                                  
├── utils                         
│   ├── __init__.py              
│   ├── calculate_gait.py         # Used for calculating robot gait.
│   ├── helpers.py                # Processing parameters, processing paths
│   ├── logger.py                 
│   ├── task_registry.py          # Set up the environment, robots, and training parameters.
│   ├── terrain.py                # Set the terrain.
│   └── math.py                   # math tools 
│                                  
└── README.md                                         
