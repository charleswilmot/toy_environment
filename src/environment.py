import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import jsonpybox2d as json2d
import numpy as np
import tactile_map as tm
import pid
from PIL import Image, ImageDraw
import pickle


def discretize(arr, mini, maxi, n):
    discrete = np.tile(np.linspace(mini, maxi, n), list(arr.shape) + [1])
    discrete -= np.expand_dims(arr, -1)
    discrete = np.cos(np.pi * discrete / (maxi - mini) - np.pi * (maxi + mini) / (maxi - mini)) ** 200
    return discrete


def custom_mod_2_pi(x, center=0.0):
    return ((x + np.pi - center) % (2 * np.pi)) - (np.pi - center)


class Environment(object):
    """ 2D physics using box2d and a json conf file
    """
    def __init__(self, world_file, skin_order, skin_resolution, xlim, ylim, dpi, env_step_length,
                 dt=1 / 120.0, n_discrete=32):
        """

            :param world_file: the json file from which all objects are created
            :type world_file: string

            :param dt: the amount of time to simulate, this should not vary.
            :type dt: float

            :param pos_iters: for the velocity constraint solver.
            :type pos_iters: int

            :param vel_iters: for the position constraint solver.
            :type vel_iters: int

        """
        world, bodies, joints = json2d.createWorldFromJson(world_file)
        self._count = 0
        self.dt = dt
        self._vel_iters = 6
        self._pos_iters = 2
        self._dpi = dpi
        self.env_step_length = env_step_length
        self._n_discrete = n_discrete
        self.world = world
        self.bodies = bodies
        self.to_name = {self.bodies[name]: name for name in self.bodies}
        self.contact_logs = {}
        self.joints = joints
        self.used_bodies = {k: bodies[k] for k in bodies if k in [a for a, b in skin_order]}
        self.joint_pids = {key: pid.PID(dt=self.dt)
                           for key in self.joints}
        self._joint_keys = [key for key in sorted(self.joints)]
        self._joint_keys.sort()
        self._buf_positions = np.zeros(len(self.joints))
        self._buf_target_positions = np.zeros(len(self.joints))
        self._buf_speeds = np.zeros(len(self.joints))
        self.skin = tm.Skin(self.bodies, skin_order, skin_resolution)
        self._joints_in_position_mode = set()
        tactile_bodies_names = set([body_name for body_name, edge in skin_order])
        self.renderer = Renderer(self.bodies, xlim, ylim, tactile_bodies_names=tactile_bodies_names, dpi=dpi)
        self._computed_vision = False
        self._computed_tactile = False
        self._computed_positions = False
        self._computed_discrete_positions = False
        self._computed_speeds = False
        self._computed_discrete_speeds = False
        self._computed_target_positions = False
        self._computed_discrete_target_positions = False

    def log_contacts(self):
        contacts = {body:
                    [self.to_name[ce.other] for ce in self.bodies[body].contacts
                     if ce.contact.touching] for body in self.used_bodies}
        contacts = {body: contacts[body] for body in contacts if len(contacts[body]) > 0}
        if len(contacts) != 0:
            if self._count in self.contact_logs:
                for body in contacts:
                    if body in self.contact_logs[self._count]:
                        for other in contacts[body]:
                            if other not in self.contact_logs[self._count][body]:
                                self.contact_logs[self._count][body].append(other)
                    else:
                        self.contact_logs[self._count][body] = contacts[body]
            else:
                self.contact_logs[self._count] = contacts

    def save_contact_logs(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.contact_logs, f)

    def set_speeds(self, speeds):
        for key in speeds:
            if key in self.joints:
                if key in self._joints_in_position_mode:
                    self._joints_in_position_mode.remove(key)
                self.joints[key].motorSpeed = np.float64(speeds[key])

    def set_positions(self, positions):
        threshold = 0.05
        for i, key in enumerate(self._joint_keys):
            if key in positions:
                self._joints_in_position_mode.add(key)
                pos = positions[key]
                current = self.joints[key].angle
                if self.joints[key].limitEnabled:
                    min_lim, max_lim = self.joints[key].limits
                    if pos < min_lim + threshold:
                        pos = min_lim - threshold * threshold / (pos - 2 * threshold - min_lim)
                    elif pos > max_lim - threshold:
                        pos = max_lim - threshold * threshold / (pos + 2 * threshold - max_lim)
                else:
                    pos = custom_mod_2_pi(pos, center=current)
                self._buf_target_positions[i] = pos
                self.joint_pids[key].setpoint = pos
        self._computed_discrete_target_positions = False

    def step(self):
        for key in self._joints_in_position_mode:
            self.joint_pids[key].step(self.joints[key].angle)
            self.joints[key].motorSpeed = np.clip(self.joint_pids[key].output, -np.pi, np.pi)
        self.log_contacts()
        self.world.Step(self.dt, self._vel_iters, self._pos_iters)
        self._computed_vision = False
        self._computed_tactile = False
        self._computed_positions = False
        self._computed_discrete_positions = False
        self._computed_speeds = False
        self._computed_discrete_speeds = False
        self._computed_target_positions = False
        self._computed_discrete_target_positions = False

    def env_step(self):
        for i in range(self.env_step_length):
            self.step()
        self._count += 1

    def _get_state_vision(self):
        if self._computed_vision:
            return self._buf_vision
        else:
            self._buf_vision = self.renderer.step()
            self._computed_vision = True
            return self._buf_vision

    def _get_state_tactile(self):
        if self._computed_tactile:
            return self._buf_tactile
        else:
            self._buf_tactile = self.skin.compute_map()
            self._computed_tactile = True
            return self._buf_tactile

    def _get_state_positions_old(self):
        if self._computed_positions:
            return self._buf_positions
        else:
            for i, key in enumerate(self._joint_keys):
                self._buf_positions[i] = self.joints[key].angle
            self._buf_positions %= 2 * np.pi
            self._buf_positions -= np.pi
            self._computed_positions = True
            return self._buf_positions

    def _get_state_positions(self):
        if self._computed_positions:
            return self._buf_positions
        else:
            for i, key in enumerate(self._joint_keys):
                self._buf_positions[i] = self.joints[key].angle
            self._computed_positions = True
            return self._buf_positions

    def _get_state_discrete_positions(self):
        if self._computed_discrete_positions:
            return self._buf_discrete_positions
        else:
            self._buf_discrete_positions = discretize(self.positions, -np.pi, np.pi, self._n_discrete)
            self._computed_discrete_positions = True
            return self._buf_discrete_positions

    def _get_state_speeds(self):
        if self._computed_speeds:
            return self._buf_speeds
        else:
            for i, key in enumerate(self._joint_keys):
                self._buf_speeds[i] = self.joints[key].speed
            self._computed_speeds = True
            return self._buf_speeds

    def _get_state_discrete_speeds(self):
        if self._computed_discrete_speeds:
            return self._buf_discrete_speeds
        else:
            self._buf_discrete_speeds = discretize(self.speeds, -np.pi, np.pi, self._n_discrete)
            self._computed_discrete_speeds = True
            return self._buf_discrete_speeds

    def _get_state_target_positions(self):
        return self._buf_target_positions

    def _get_state_discrete_target_positions(self):
        if self._computed_discrete_target_positions:
            return self._buf_discrete_target_positions
        else:
            self._buf_discrete_target_positions = discretize(self.target_positions, -np.pi, np.pi, self._n_discrete)
            self._computed_discrete_target_positions = True
            return self._buf_discrete_target_positions

    def _get_state(self):
        vision = self.vision
        positions = self.positions
        speeds = self.speeds
        tactile_map = self.tactile
        return vision, positions, speeds, tactile_map

    state = property(_get_state)
    positions = property(_get_state_positions)
    discrete_positions = property(_get_state_discrete_positions)
    speeds = property(_get_state_speeds)
    discrete_speeds = property(_get_state_discrete_speeds)
    target_positions = property(_get_state_target_positions)
    discrete_target_positions = property(_get_state_discrete_target_positions)
    vision = property(_get_state_vision)
    tactile = property(_get_state_tactile)


class Renderer:
    def __init__(self, bodies, xlim, ylim, dpi, tactile_bodies_names=[]):
        self.bodies = bodies
        self.tactile_bodies_names = tactile_bodies_names
        self._x_lim = xlim
        self._y_lim = ylim
        self._max_x = int(dpi * (xlim[1] - xlim[0]))
        self._max_y = int(dpi * (ylim[1] - ylim[0]))
        self.shape = [self._max_x, self._max_y]
        self.dpi = dpi
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = Image.new('RGB', self.shape, (255, 255, 255))
        self.draw = ImageDraw.Draw(self.buffer)

    def point_to_pix(self, point):
        x, y = point
        X = self._max_x * (x - self._x_lim[0]) / (self._x_lim[1] - self._x_lim[0])
        Y = self._max_y * (y - self._y_lim[0]) / (self._y_lim[1] - self._y_lim[0])
        return X, Y

    def step(self):
        self.reset_buffer()
        for key in self.bodies:
            body = self.bodies[key]
            touching = [ce.contact.touching for ce in body.contacts if ce.contact.touching]
            vercs = np.vstack(body.fixtures[0].shape.vertices)
            data = [self.point_to_pix(body.GetWorldPoint(x)) for x in vercs]
            color = (255, 0, 0) if len(touching) > 0 and key in self.tactile_bodies_names else (0, 0, 255)
            self.draw.polygon(data, fill=color)
        return np.asarray(self.buffer)


if __name__ == "__main__":
    import viewer

    win = viewer.VisionJointsSkinWindow()

    skin_order = [
        ("Arm1_Left", 0),
        ("Arm2_Left", 0),
        ("Arm2_Left", 1),
        ("Arm2_Left", 2),
        ("Arm1_Left", 2),
        ("Arm1_Right", 0),
        ("Arm2_Right", 0),
        ("Arm2_Right", 1),
        ("Arm2_Right", 2),
        ("Arm1_Right", 2)]
    skin_resolution = 12
    xlim = [-20.5, 20.5]
    ylim = [-13.5, 13.5]
    env = Environment("../models/two_arms.json", skin_order, skin_resolution, xlim, ylim, dpi=10, dt=1 / 150.0)

    for i in range(1000):
        actions = {
            "Arm1_to_Arm2_Left": np.random.uniform(-2.3, 2.3),
            "Ground_to_Arm1_Left": np.random.uniform(-3.14, 3.14),
            "Arm1_to_Arm2_Right": np.random.uniform(-2.3, 2.3),
            "Ground_to_Arm1_Right": np.random.uniform(-3.14, 3.14)
        }
        env.set_positions(actions)
        for j in range(1000):
            if j % 100 == 0:
                win.update(*env.state)
            env.step()
