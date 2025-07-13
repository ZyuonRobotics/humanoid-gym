from itertools import product
from pathlib import Path
from humanoid.envs.custom.humanoid_config import XBotLCfg, XBotLCfgPPO


class ZhaplinArmlessCfg(XBotLCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(XBotLCfg.env):
        # change the observation dim
        num_active_dofs = 11
        num_passive_dofs = 0
        num_commands = 5

        frame_stack = 15
        c_frame_stack = 3

        obs_names = ["command_input", "dof_pos", "dof_vel", "actions", "base_ang_vel", "base_euler_xyz"]
        privileged_obs_names = [
            "command_input", "dof_pos", "dof_vel", "actions", "target_dof_pos",
            "base_lin_vel", "base_ang_vel", "base_euler_xyz", "rand_push_force", "rand_push_force",
            "friction_coeffs", "restitution_coeffs", "base_mass_coeffs", "base_com_coeffs",
            "joint_friction_coeffs", "joint_armature_coeffs", "joint_pos_biases",
            "joint_kp_coeffs", "joint_kd_coeffs",
            "stance_mask", "contact_mask", "base_euler_bias"
        ]

        num_envs = 4096
        episode_length_s = 24     # episode length in seconds
        use_ref_actions = False   # speed up training by using reference actions

    class safety(XBotLCfg.safety):
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 1.0

    class asset(XBotLCfg.asset):
        file = file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/Zhaplinarmless/robot.xml'
        name = "zhaplin"
        foot_names = ["right-ankle-pitch", "left-ankle-pitch"]
        knee_names = ["right-knee-pitch", "left-knee-pitch"]

        terminate_after_contacts_on = ['torso']
        penalize_contacts_on = ["torso"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(XBotLCfg.terrain):
        mesh_type = 'plane'
        # mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.

    class noise(XBotLCfg.noise):
        add_noise = True
        noise_level = 0.6    # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            base_ang_vel = 0.1
            base_lin_vel = 0.05
            base_euler = 0.03
            height_measurements = 0.1

    class init_state(XBotLCfg.init_state):
        pos = [0.0, 0.0, 0.75]
        default_joint_angles = { 
        "waist-yaw":0.,
        "left-hip-pitch":0.,
        "left-hip-roll":0.,
        "left-hip-yaw":0.,
        "left-knee-pitch":0.,
        "left-ankle-pitch":0.,
        "right-hip-pitch" :0.,
        "right-hip-roll":0.,
        "right-hip-yaw":0.,
        "right-knee-pitch":0.,
        "right-ankle-pitch":0.,
        }

    class control(XBotLCfg.control):
        stiffness = { 
        "waist-yaw":30.,
        "left-hip-pitch":30.,
        "left-hip-roll":30.,
        "left-hip-yaw":30.,
        "left-knee-pitch":30.,
        "left-ankle-pitch":30.,
        "right-hip-pitch" :30.,
        "right-hip-roll":30.,
        "right-hip-yaw":30.,
        "right-knee-pitch":30.,
        "right-ankle-pitch":30.,
        }

        damping = { 
        "waist-yaw":3.,
        "left-hip-pitch":3.,
        "left-hip-roll":3.,
        "left-hip-yaw":3.,
        "left-knee-pitch":3.,
        "left-ankle-pitch":3.,
        "right-hip-pitch" :3.,
        "right-hip-roll":3.,
        "right-hip-yaw":3.,
        "right-knee-pitch":3.,
        "right-ankle-pitch":3.,
        }

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 100hz

    class sim(XBotLCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1
        up_axis = 1  # 0 is y, 1 is z

        class physx(XBotLCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand(XBotLCfg.domain_rand):
        pass

    class commands(XBotLCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges(XBotLCfg.commands.ranges):
            lin_vel_x = [-0.3, 0.6]   # min max [m/s]
            lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3] # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards(XBotLCfg.rewards):
        base_height_target = 0.75
        min_dist = 0.2
        max_dist = 0.5
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.3    # rad
        target_feet_height = 0.06        # m
        cycle_time = 0.64                # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5
        max_contact_force = 700  # Forces above this value are penalized

        class scales(XBotLCfg.rewards.scales):
            # reference motion tracking
            joint_pos = 1.6
            feet_clearance = 1.
            feet_contact_number = 1.2
            # gait
            feet_air_time = 1.
            foot_slip = -0.05
            feet_distance = 0.2
            knee_distance = 0.2
            # contact
            feet_contact_forces = -0.01
            # vel tracking
            tracking_lin_vel = 1.2
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            low_speed = 0.2
            track_vel_hard = 0.5
            # base pos
            default_joint_pos = 0.5
            orientation = 1.
            base_height = 0.2
            base_acc = 0.2
            # energy
            action_smoothness = -0.002
            torques = -1e-5
            dof_vel = -5e-4
            dof_acc = -1e-7
            collision = -1.

    class normalization(XBotLCfg.normalization):
        clip_observations = 18.
        clip_actions = 18.

class ZhaplinArmlessCfgPPO(XBotLCfgPPO):
    seed = 5
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

    class policy(XBotLCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(XBotLCfgPPO.algorithm):
        entropy_coef = 0.001
        symm_loss_coef = 0.3
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner(XBotLCfgPPO.runner):
        num_steps_per_env = 60  # per iteration
        max_iterations = 30001  # number of policy updates

        # logging
        save_interval = 50  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'zhaplin_armless_ppo'
