# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

from itertools import product

from humanoid.envs.custom.humanoid_config import XBotLCfg, XBotLCfgPPO


class G1Cfg(XBotLCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(XBotLCfg.env):
        # change the observation dim
        num_active_dofs = 12
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
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/mjcf/12dof.xml'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/urdf/12dof.urdf'

        name = "g1"
        foot_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
        knee_names = ["left_knee_joint", "right_knee_joint"]

        terminate_after_contacts_on = ["pelvis"]
        penalize_contacts_on = ["pelvis"]
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
        pos = [0.0, 0.0, 0.793]

        default_joint_angles = {
            # "waist_yaw_joint": 0,
            # "waist_roll_joint": 0,
            # "waist_pitch_joint": 0
        }
        for lr, name in product(['left', 'right'], ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]):
            default_joint_angles[f'{lr}_{name}_joint'] = 0
        # for lr, name in product(['left', 'right'], ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]):
        #     default_joint_angles[f'{lr}_{name}_joint'] = 0.

    class control(XBotLCfg.control):
        stiffness = {
            # "waist_yaw_joint": 40,
            # "waist_roll_joint": 40,
            # "waist_pitch_joint": 40
        }
        for lr, name, value in product(
                ['left', 'right'],
                ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"],
                [100., 100., 100., 140., 40., 40.]
        ):
            stiffness[f'{lr}_{name}_joint'] = value
        # for lr, name, value in product(
        #         ['left', 'right'],
        #         ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"],
        #         [40.]*7
        # ):
        #     stiffness[f'{lr}_{name}_joint'] = value

        damping = {
            # "waist_yaw_joint": 2,
            # "waist_roll_joint": 2,
            # "waist_pitch_joint": 2
        }
        for lr, name, value in product(
                ['left', 'right'],
                ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"],
                [2., 2, 2, 4, 2, 2]
        ):
            damping[f'{lr}_{name}_joint'] = value
        # for lr, name, value in product(
        #         ['left', 'right'],
        #         ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll",  "wrist_pitch", "wrist_yaw"],
        #         [2.] * 7
        # ):
        #     damping[f'{lr}_{name}_joint'] = value

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
        base_height_target = 0.793
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


class G1CfgPPO(XBotLCfgPPO):
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
        experiment_name = 'g1_ppo'