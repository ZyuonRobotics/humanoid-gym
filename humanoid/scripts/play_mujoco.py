import argparse
from os import path
from collections import deque
import time
import curses
import math

import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime
from scipy.spatial.transform import Rotation

BASE_PATH = path.dirname(path.dirname(path.dirname(__file__)))
KUAVO_MJCF_PATH = path.join(BASE_PATH, "resources", "robots", "XBot", "mjcf", "robot.xml")

P_GAINS = [200, 200, 350, 350, 15, 15, 200, 200, 350, 350, 15, 15]
D_GAINS = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

DEFAULT_HEIGHT = 0.9
DEFAULT_JOINT_POS = [0] * 12

PERIOD_LENGTH = 0.64
INPUT_LIST =["command", "dof_pos", "dof_vel", "actions", "base_ang_vel", "base_euler_xyz"]
def get_local_value(v, quat):
    r = Rotation.from_quat(quat)
    return r.apply(v, inverse=True)

class PlayMujoco:
    def __init__(
            self,
            onnx_path,
            change_period=False,
            frame_stack=15,
            frame_stack_skip=1,
            dof_num=12,
            robot_xml_path=KUAVO_MJCF_PATH,
            control_freq=10,
            p_gains=None,
            d_gains=None,
            default_height=DEFAULT_HEIGHT,
            default_dof_pos=None,
            input_list=None
    ):
        self.onnx_path = onnx_path
        self.change_period = change_period
        self.frame_stack = frame_stack
        self.frame_stack_skip = frame_stack_skip
        self.action_num = dof_num
        self.robot_xml_path = robot_xml_path
        self.control_freq = control_freq
        self.p_gains = np.array(P_GAINS) if p_gains is None else p_gains
        self.d_gains = np.array(D_GAINS) if d_gains is None else d_gains
        self.default_height = default_height
        self.default_dof_pos = np.array(DEFAULT_JOINT_POS) if default_dof_pos is None else default_dof_pos
        self.input_list = input_list if input_list is not None else INPUT_LIST

        self.input_info = {}
        for i, name in enumerate(self.input_list):
            if name == "command":
                self.input_info[name] = 5
            elif name in ["dof_pos", "dof_vel", "actions"]:
                self.input_info[name] = dof_num
            elif name in ["base_ang_vel", "base_euler_xyz", "base_lin_acc"]:
                self.input_info[name] = 3
            else:
                raise ValueError(f"Invalid input name: {name}")

        self.ort_session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        self.obs_history = deque(maxlen=frame_stack)
        self.real_frame_stack = int(frame_stack / frame_stack_skip)
        for _ in range(frame_stack):
            self.obs_history.append(np.zeros(sum(self.input_info.values())))

        self.model = mujoco.MjModel.from_xml_path(self.robot_xml_path)
        self.torque_limit = np.abs(self.model.actuator_ctrlrange[:, 0])

        self.data = mujoco.MjData(self.model)
        self.data.qpos[:7] = [0., 0., self.default_height, 1., 0., 0., 0.]
        self.data.qpos[7:] = self.default_dof_pos.copy()
        self.data.qvel[:] = 0
        self.torque = np.zeros(dof_num)

        self.time_s = 0
        self.pre_action = np.zeros(dof_num)
        self.base_lin_acc = np.zeros(3)

        self.vel_x, self.vel_y, self.vel_yaw = 0.0, 0, 0
        self.standing = 0

        self.actions = np.zeros(dof_num)

    def change_command(self, x, y, yaw):
        pre_period_length = self.period_length()
        self.vel_x, self.vel_y, self.vel_yaw = x, y, yaw
        new_period_length = self.period_length()

        self.time_s = (self.time_s % pre_period_length) / pre_period_length * new_period_length

    def update_command(self, stdscr):
        stdscr.nodelay(True)
        key = stdscr.getch()
        if key == ord('w'):
            self.standing = 0
            self.vel_x += 0.1
        elif key == ord('s'):
            self.standing = 0
            self.vel_x -= 0.1
        elif key == ord('a'):
            self.standing = 0
            self.vel_y -= 0.1
        elif key == ord('d'):
            self.standing = 0
            self.vel_y += 0.1
        elif key == ord('q'):
            self.standing = 0
            self.vel_yaw -= 0.1
        elif key == ord('e'):
            self.standing = 0
            self.vel_yaw += 0.1
        elif key == ord(' '):
            self.standing = 1
            self.vel_x, self.vel_y, self.vel_yaw = 0, 0, 0
        elif key == ord('m'):
            self.standing = 0
            self.vel_x, self.vel_y, self.vel_yaw = 0, 0, 0

    def show_command(self, stdscr):
        stdscr.clear()
        stdscr.addstr(0, 0, f'Current velocity: x={self.vel_x}, y={self.vel_y}, yaw={self.vel_yaw}, '
                            f'period={self.period_length}, phase={self.phase}')
        stdscr.refresh()

    def act(self):
        self.obs_history.append(self.get_obs_now())

        obs_buf_all = np.stack([self.obs_history[i] for i in range(self.frame_stack)], axis=0)
        obs_buf_all = obs_buf_all.reshape(self.real_frame_stack, self.frame_stack_skip, -1).astype(np.float32)
        obs_buf_all = np.clip(obs_buf_all, -18., 18.)

        input_name = self.ort_session.get_inputs()[0].name
        outputs = self.ort_session.run(None, {input_name: obs_buf_all[:, -1].reshape(1, -1)})
        self.actions = np.clip(outputs[0][0], -18., 18.)

    @property
    def period_length(self):
        if self.change_period:
            vel = (self.vel_x ** 2 + self.vel_y ** 2) ** 0.5
            return 1. / (0.56 * vel + 0.39) if vel > 0.0 else 0
        else:
            return PERIOD_LENGTH

    @property
    def phase(self):
        if self.standing == 1:
            return 0
        else:
            return self.time_s / self.period_length

    def get_obs_now(self):
        phase = 2 * math.pi * self.phase
        commands = np.array([math.sin(phase), math.cos(phase), self.vel_x, self.vel_y, self.vel_yaw])
        dof_pos = self.data.qpos[7:]
        dof_vel = self.data.qvel[6:]

        quat = self.data.qpos[3:7][[1, 2, 3, 0]]
        r = Rotation.from_quat(quat)
        base_euler_xyz = r.as_euler("xyz")
        base_ang_vel = self.data.qvel[3:6]

        obs_now = np.concatenate([
            commands, dof_pos, dof_vel,
            self.pre_action,
            base_ang_vel, base_euler_xyz,
        ]).copy()

        return obs_now

    def play(self, stdscr):
        curses.curs_set(0)

        dof_pos_list = []
        torque_list = []
        acc_list = []

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                self.update_command(stdscr)
                self.show_command(stdscr)

                start_time = time.time()
                self.act()

                for i in range(self.control_freq):
                    self.torque = self.p_gains * (self.actions * 0.25 + self.default_dof_pos - self.data.qpos[7:]) - self.d_gains * self.data.qvel[6:]
                    self.torque = np.clip(self.torque, -self.torque_limit, self.torque_limit)

                    self.data.ctrl[:] = self.torque

                    mujoco.mj_step(self.model, self.data, 1)

                viewer.sync()

                end_time = time.time()
                time.sleep(max(0, 0.01 - (end_time - start_time)))
                self.pre_action = self.actions.copy()

                dof_pos_list.append(self.data.qpos[7:].copy())
                torque_list.append(self.actions.copy())
                acc_list.append(self.base_lin_acc.copy())

                self.time_s += 0.01


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_path", type=str)
    args = parser.parse_args()

    change_period = "amass" in args.onnx_path

    play_mujoco = PlayMujoco(onnx_path=args.onnx_path, change_period=change_period)
    curses.wrapper(play_mujoco.play)
    # play_mujoco.play(None)
