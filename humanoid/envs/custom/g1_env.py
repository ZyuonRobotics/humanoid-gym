from humanoid.envs.base.legged_robot_config import LeggedRobotCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import torch
from humanoid.envs import XBotLFreeEnv

from humanoid.utils.terrain import  HumanoidTerrain


class G1FreeEnv(XBotLFreeEnv):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = self.cfg.rewards.target_joint_pos_scale
        scale_2 = 2 * scale_1

        # left foot stance phase set to default joint pos
        sin_pos_l[sin_pos_l > -0.1] = 0
        self.ref_dof_pos[:, 0] = sin_pos_l * scale_1
        self.ref_dof_pos[:, 3] = -sin_pos_l * scale_2
        self.ref_dof_pos[:, 4] = sin_pos_l * scale_1
        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r < 0.1] = 0
        self.ref_dof_pos[:, 5] = -sin_pos_r * scale_1
        self.ref_dof_pos[:, 8] = sin_pos_r * scale_2
        self.ref_dof_pos[:, 9] = -sin_pos_r * scale_1

        self.ref_action = 2 * self.ref_dof_pos

    def get_symm_dof(self, value):
        res = value.clone()
        axis = self.get_dof_axis()
        yaw_or_roll = axis[:, 0].abs().bool() | axis[:, 2].abs().bool()

        res[:, :12] = torch.roll(res[:, :12], shifts=6, dims=-1)
        res[:, 15:29] = torch.roll(res[:, 15:29], shifts=7, dims=-1)
        res[:, yaw_or_roll] *= -1
        return res

# ================================================ Rewards ================================================== #
    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.dof_pos - self.default_joint_pd_target
        left_yaw_roll = joint_diff[:, [1, 2, 5]]
        right_yaw_roll = joint_diff[:, [7, 8, 11]]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)

    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos - pos_target
        r = torch.exp(-1 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        return r