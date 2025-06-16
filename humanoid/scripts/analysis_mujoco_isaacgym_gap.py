import argparse
from os import path


import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

BASE_PATH = path.dirname(path.dirname(path.dirname(__file__)))
KUAVO_MJCF_PATH = path.join(BASE_PATH, "resources", "robots", "g1", "mjcf", "12dof.xml")

DOF_NUM = 12

P_GAINS = [100., 100., 100., 140., 40., 40.] * 2
D_GAINS =  [2., 2, 2, 4, 2, 2] * 2

DEFAULT_HEIGHT = 0.75
DEFAULT_JOINT_POS = [0] * DOF_NUM

def get_local_value(v, quat):
    r = Rotation.from_quat(quat)
    return r.apply(v, inverse=True)

class PlayMujoco:
    def __init__(
            self,
            npz_path,
            dof_num=DOF_NUM,
            robot_xml_path=KUAVO_MJCF_PATH,
            control_freq=10,
            p_gains=None,
            d_gains=None,
            default_height=DEFAULT_HEIGHT,
            default_dof_pos=None,
    ):
        self.npz_path = npz_path
        self.action_num = dof_num
        self.robot_xml_path = robot_xml_path
        self.control_freq = control_freq
        self.p_gains = np.array(P_GAINS) if p_gains is None else p_gains
        self.d_gains = np.array(D_GAINS) if d_gains is None else d_gains
        self.default_height = default_height
        self.default_dof_pos = np.array(DEFAULT_JOINT_POS) if default_dof_pos is None else default_dof_pos

        self.npz = np.load(self.npz_path, allow_pickle=True)
        self.isaacgym_qpos = self.npz["qpos"][:, 0]
        self.isaacgym_qvel = self.npz["qvel"][:, 0]
        self.isaacgym_action = self.npz["action"][:, 0]
        self.isaacgym_length = self.isaacgym_qpos.shape[0]

        self.model = mujoco.MjModel.from_xml_path(self.robot_xml_path)
        self.torque_limit = np.abs(self.model.actuator_ctrlrange[:, 0])

        self.data = mujoco.MjData(self.model)
        self.data.qpos[:7] = [0., 0., self.default_height, 1., 0., 0., 0.]
        self.data.qpos[7:] = self.default_dof_pos.copy()
        self.data.qvel[:] = 0

        self.mujoco_qpos = self.isaacgym_qpos.copy()
        self.mujoco_qvel = self.isaacgym_qvel.copy()

    def get_figure(self, n, dpi=100):
        sqrt_n = np.sqrt(n)
        cols = int(np.ceil(sqrt_n))
        rows = int(np.ceil(n / cols))

        while (cols / rows) < 1 and cols * (rows - 1) >= n:
            rows -= 1

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), dpi=dpi)
        fig.tight_layout(pad=3.0)

        if isinstance(axes, np.ndarray):
            axes_flat = axes.ravel()
        else:
            axes_flat = np.array([axes])

        for i in range(n, len(axes_flat)):
            axes_flat[i].axis('off')

        for i, ax in enumerate(axes_flat[:n]):
            ax.set_title(f'Dim {i}', fontsize=10)
            ax.tick_params(labelsize=8)

        return fig, axes_flat[:n] if n > 1 else axes_flat[0]

    def play(self):
        for i in range(self.isaacgym_length - 1):
            self.data.qpos[:] = self.isaacgym_qpos[i]
            self.data.qvel[:] = self.isaacgym_qvel[i]
            mujoco.mj_forward(self.model, self.data)

            for _ in range(self.control_freq):
                self.torque = (self.p_gains * (self.isaacgym_action[i] * 0.25 - self.data.qpos[7:])
                               - self.d_gains * self.data.qvel[6:])
                self.torque = np.clip(self.torque, -self.torque_limit, self.torque_limit)

                self.data.ctrl[:] = self.torque

                mujoco.mj_step(self.model, self.data, 1)

            self.mujoco_qpos[i+1] = self.data.qpos
            self.mujoco_qvel[i+1] = self.data.qvel

    def draw_all_qpos(self):
        fig, axes = self.get_figure(self.mujoco_qpos.shape[1])
        for i in range(self.mujoco_qpos.shape[1]):
            axes[i].plot(self.isaacgym_qpos[:, i], label="isaacgym")
            axes[i].plot(self.mujoco_qpos[:, i], label="mujoco")
            # axes[i].plot(self.isaacgym_action[:, i] * 0.25, label="action")
            axes[i].legend()
        plt.show()

    def draw_all_qvel(self):
        fig, axes = self.get_figure(self.mujoco_qvel.shape[1])
        for i in range(self.mujoco_qvel.shape[1]):
            axes[i].plot(self.isaacgym_qvel[:, i], label="isaacgym")
            axes[i].plot(self.mujoco_qvel[:, i], label="mujoco")
            axes[i].legend()
        plt.show()

    def draw_qpos(self, dim=0):
        plt.plot(self.isaacgym_qpos[:, dim], label="isaacgym")
        plt.plot(self.mujoco_qpos[:, dim], label="mujoco")
        # plt.plot(self.isaacgym_action[:, dim] * 0.25, label="action")
        plt.legend()
        plt.show()

    def draw_qvel(self, dim=0):
        plt.plot(self.isaacgym_qvel[:, dim], label="isaacgym")
        plt.plot(self.mujoco_qvel[:, dim], label="mujoco")
        plt.legend()
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", type=str)
    args = parser.parse_args()

    play_mujoco = PlayMujoco(npz_path=args.npz_path)
    play_mujoco.play()

    # play_mujoco.draw_all_qpos()
    play_mujoco.draw_all_qvel()
    # play_mujoco.draw_qpos(7)
    # play_mujoco.draw_qvel(6)
