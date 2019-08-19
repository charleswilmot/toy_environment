import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf


plt.ion()


class Viewer:
    def __init__(self, path, fps=25):
        self.path = path + "/"
        self.fps = fps
        self.data = database.Database(self.path)
        self.window = Window()

    def __call__(self):
        t0 = time.time()
        i = 0
        while i < len(self.data):
            self.display(i)
            t1 = time.time()
            delta = t1 - t0
            i = int(self.fps * delta)

    def display(self, i):
        vision, proprioception, tactile_map = self.data[i]
        self.window.update(vision, proprioception, tactile_map)


class VisionIAX:
    def __init__(self, ax):
        self.ax = ax
        self._lim = [[0, 255]]
        self._axes_initialized = False

    def __initaxes__(self, vision):
        self._image = self.ax.imshow(vision)
        self._image.set_clim(*self._lim)
        self._image.axes.get_xaxis().set_visible(False)
        self._image.axes.get_yaxis().set_visible(False)
        self._axes_initialized = True

    def set_lim(self, *lim):
        self._lim = lim
        if self._axes_initialized:
            self._image.set_clim(*lim)

    def __call__(self, vision):
        if self._axes_initialized:
            self._image.set_data(vision)
        else:
            self.__initaxes__(vision)


class TactileIAX:
    def __init__(self, ax):
        self.ax = ax
        self._lim = [[-0.1, 1.1]]
        self._axes_initialized = False

    def __initaxes__(self, tactile_maps):
        self._lines = list(self.ax.plot(tactile_maps.T))
        self.ax.set_ylim(*self._lim)
        self.ax.axes.get_yaxis().set_ticks([0, 1])
        self.ax.set_title("Skin sensor")
        self._axes_initialized = True

    def set_lim(self, *lim):
        self._lim = lim
        if self._axes_initialized:
            self.ax.set_ylim(*lim)

    def __call__(self, tactile_maps):
        if self._axes_initialized:
            if tactile_maps.ndim == 1:
                self._lines[0].set_ydata(tactile_maps)
            else:
                for tm, l in zip(tactile_maps, self._lines):
                    l.set_ydata(tm)
        else:
            self.__initaxes__(tactile_maps)


class JointsIAX:
    def __init__(self, ax):
        self.ax = ax
        self._lim = [[-4, 4]]
        self._axes_initialized = False

    def __initaxes__(self, positions, speeds):
        self._angle_line, = self.ax.plot(positions, "o")
        self._speed_line, = self.ax.plot(speeds, "o")
        self.ax.set_ylim(*self._lim)
        self.ax.axes.get_xaxis().set_ticks(range(speeds.shape[0]))
        self.ax.set_title("Joints speed/position")
        self._axes_initialized = True

    def set_lim(self, *lim):
        self._lim = lim
        if self._axes_initialized:
            self.ax.set_ylim(*lim)

    def __call__(self, positions, speeds):
        if self._axes_initialized:
            self._speed_line.set_ydata(positions)
            self._angle_line.set_ydata(speeds)
        else:
            self.__initaxes__(positions, speeds)


class OnlineReturnIAX:
    def __init__(self, ax, discount_factor, return_lookback=100, lim=[[-5, 5]]):
        self.ax = ax
        self._discount_factor = discount_factor
        self._return_lookback = return_lookback
        self._estimated_true_return = np.zeros(self._return_lookback)
        self._predicted_return = np.zeros(self._return_lookback)
        self._rewards = np.zeros(self._return_lookback)
        self._lim = lim
        self._axes_initialized = False

    def __initaxes__(self):
        X = np.arange(1 - self._return_lookback, 1)
        self._predicted_line = self.ax.plot(X, self._predicted_return, "--")[0]
        self._estimated_true_line = self.ax.plot(X, self._estimated_true_return, "--")[0]
        self.ax.axvline(0, color="r")
        self.ax.set_ylim(*self._lim)
        self.ax.set_title("Return")
        self._axes_initialized = True

    def set_lim(self, *lim):
        self._lim = lim
        if self._axes_initialized:
            self.ax.set_ylim(*lim)

    def set_range(self, *rnge):
        self._range = rnge
        if self._axes_initialized:
            for line in self._lines:
                line.set_xdata(np.arange(*rnge))

    def _update_buffer(self, current_reward, predicted_return):
        self._predicted_return[:-1] = self._predicted_return[1:]
        self._predicted_return[-1] = predicted_return
        self._rewards[:-1] = self._rewards[1:]
        self._rewards[-1] = current_reward
        prev = predicted_return
        self._estimated_true_return[-1] = prev
        for i in np.arange(self._return_lookback - 2, -1, -1):
            self._estimated_true_return[i] = prev * self._discount_factor + self._rewards[i]
            prev = self._estimated_true_return[i]

    def __call__(self, current_reward, predicted_return):
        self._update_buffer(current_reward, predicted_return)
        if self._axes_initialized:
            self._estimated_true_line.set_ydata(self._estimated_true_return)
            self._predicted_line.set_ydata(self._predicted_return)
        else:
            self.__initaxes__()


class ReturnIAX:
    def __init__(self, ax, rturn):
        self.ax = ax
        self._lim = [[0, 0.6]]
        self._range = (-50, 50)
        self._axes_initialized = False
        self._return = rturn.reshape((rturn.shape[0], -1))
        self._maxi = self._return.shape[0]
        self._n_lines = self._return.shape[-1]

    def set_return(self, rturn):
        self._return = rturn.reshape((rturn.shape[0], -1))
        if self._return.shape[-1] != self._n_lines:
            raise ValueError("The passed return array does not have the correct amount of columns")
        self._maxi = self._return.shape[0]

    def _get_absolute_slice(self, index):
        start = max(0, index + self._range[0])
        stop = min(self._maxi, index + self._range[1])
        return slice(start, stop)

    def _get_relative_slice(self, index):
        start = max(0, - index - self._range[0])
        stop = min(self._range[1] - self._range[0], self._maxi - index + self._range[1])
        return slice(start, stop)

    def _get_data(self, index):
        data = np.zeros((self._range[1] - self._range[0], self._return.shape[1]))
        data[self._get_relative_slice(index)] = self._return[self._get_absolute_slice(index)]
        return data

    def __initaxes__(self, index):
        self._lines = self.ax.plot(np.arange(*self._range), self._get_data(index), "--")
        self.ax.axvline(0, color="r")
        self.ax.set_ylim(*self._lim)
        self.ax.set_title("Return")
        self._axes_initialized = True

    def set_lim(self, *lim):
        self._lim = lim
        if self._axes_initialized:
            self.ax.set_ylim(*lim)

    def set_range(self, *rnge):
        self._range = rnge
        if self._axes_initialized:
            for line in self._lines:
                line.set_xdata(np.arange(*rnge))

    def __call__(self, index):
        if self._axes_initialized:
            all_data = self._get_data(index)
            for i, line in enumerate(self._lines):
                data = all_data[:, i]
                line.set_ydata(data)
        else:
            self.__initaxes__(index)


class DiscreteJointPositionIAX:
    def __init__(self, ax, title):
        self.ax = ax
        self._lim = [[-0.1, 1.1]]
        self._axes_initialized = False
        self._title = title

    def __initaxes__(self, predicted_position, next_position=None, current_position=None, target_position=None):
        self._predicted_line, = self.ax.plot(predicted_position, "-")
        if next_position is not None:
            self._next_line, = self.ax.plot(next_position, "-")
        if current_position is not None:
            self._current_line, = self.ax.plot(current_position, "k--", alpha=0.6)
        if target_position is not None:
            self._target_line, = self.ax.plot(target_position, "r--", alpha=0.6)
        self.ax.set_ylim(*self._lim)
        self.ax.axes.get_xaxis().set_ticks([])
        self.ax.axes.get_yaxis().set_ticks([])
        self.ax.set_title(self._title)
        self._axes_initialized = True

    def set_title(self, title):
        self.ax.set_title(title)

    def set_lim(self, *lim):
        self._lim = lim
        if self._axes_initialized:
            self.ax.set_ylim(*lim)

    def __call__(self, predicted_position, next_position=None, current_position=None, target_position=None):
        if self._axes_initialized:
            self._predicted_line.set_ydata(predicted_position)
            if next_position is not None:
                self._next_line.set_ydata(next_position)
            if current_position is not None:
                self._current_line.set_ydata(current_position)
            if target_position is not None:
                self._target_line.set_ydata(target_position)
        else:
            self.__initaxes__(predicted_position, next_position, current_position, target_position)


class DoubleVisionWindow:
    def __init__(self):
        self.fig = plt.figure()
        self.iax_vision1 = VisionIAX(self.fig.add_subplot(211))
        self.iax_vision2 = VisionIAX(self.fig.add_subplot(212))
        self._fig_shawn = False

    def close(self):
        plt.close(self.fig)

    def set_vision_lim(self, *lim):
        self.iax_vision1.set_lim(*lim)
        self.iax_vision2.set_lim(*lim)

    def update(self, vision1, vision2):
        self.iax_vision1(vision1)
        self.iax_vision2(vision2)
        if self._fig_shawn:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        else:
            self.fig.show()
            self._fig_shawn = True


class VisionJointsSkinWindow:
    def __init__(self):
        self.fig = plt.figure()
        self.iax_vision = VisionIAX(self.fig.add_subplot(211))
        self.iax_joints = JointsIAX(self.fig.add_subplot(224))
        self.iax_tactile = TactileIAX(self.fig.add_subplot(223))
        self._fig_shawn = False

    def close(self):
        plt.close(self.fig)

    def set_vision_lim(self, *lim):
        self.iax_vision.set_lim(*lim)

    def set_tactile_lim(self, *lim):
        self.iax_tactile.set_lim(*lim)

    def set_joints_lim(self, *lim):
        self.iax_joints.set_lim(*lim)

    def update(self, vision, positions, speeds, tactile_map):
        self.iax_vision(vision)
        self.iax_joints(positions, speeds)
        self.iax_tactile(tactile_map)
        if self._fig_shawn:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        else:
            self.fig.show()
            self._fig_shawn = True


class VisionJointsReturnWindow:
    def __init__(self, rturn):
        self.fig = plt.figure()
        self.iax_vision = VisionIAX(self.fig.add_subplot(211))
        self.iax_joints = JointsIAX(self.fig.add_subplot(224))
        self.iax_return = ReturnIAX(self.fig.add_subplot(223), rturn)
        self._fig_shawn = False

    def close(self):
        plt.close(self.fig)

    def set_vision_lim(self, *lim):
        self.iax_vision.set_lim(*lim)

    def set_return_lim(self, *lim):
        self.iax_return.set_lim(*lim)

    def set_return_range(self, *rnge):
        self.iax_return.set_range(*rnge)

    def set_joints_lim(self, *lim):
        self.iax_joints.set_lim(*lim)

    def update(self, vision, positions, speeds, index):
        self.iax_vision(vision)
        self.iax_joints(positions, speeds)
        self.iax_return(index)
        if self._fig_shawn:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        else:
            self.fig.show()
            self._fig_shawn = True


class VisionDJointReturnWindow:
    def __init__(self, rturn):
        self.fig = plt.figure()
        self.iax_vision = VisionIAX(self.fig.add_subplot(511))
        left = [(5, 2, 3), (5, 2, 5), (5, 2, 7), (5, 2, 9)]
        right = [(5, 2, 4), (5, 2, 6), (5, 2, 8), (5, 2, 10)]
        jnames = ["Arm1_to_Arm2_Left", "Arm1_to_Arm2_Right", "Ground_to_Arm1_Left", "Ground_to_Arm1_Right"]
        self.iax_joints = [DiscreteJointPositionIAX(self.fig.add_subplot(*num), jname) for num, jname in zip(left, jnames)]
        self.iax_return = [ReturnIAX(self.fig.add_subplot(*num), rturn[:, i:i + 1]) for i, num in enumerate(right)]
        self._fig_shawn = False

    def close(self):
        plt.close(self.fig)

    def set_vision_lim(self, *lim):
        self.iax_vision.set_lim(*lim)

    def set_return_lim(self, *lim):
        for iax in self.iax_return:
            iax.set_lim(*lim)

    def set_return_range(self, *rnge):
        for iax in self.iax_return:
            iax.set_range(*rnge)

    def set_joint_lim(self, *lim):
        for iax in self.iax_joints:
            iax.set_lim(*lim)

    def update(self, vision, positions, targets, prevs, index):
        self.iax_vision(vision)
        for position, target, prev, iax in zip(positions, targets, prevs, self.iax_joints):
            iax(position, target, prev)
        for iax in self.iax_return:
            iax(index)
        if self._fig_shawn:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        else:
            self.fig.show()
            self._fig_shawn = True


class VisionSkinReturnWindow:
    def __init__(self, rturn):
        self.fig = plt.figure()
        self.iax_vision = VisionIAX(self.fig.add_subplot(211))
        self.iax_tactile_map = TactileIAX(self.fig.add_subplot(223))
        self.iax_return = ReturnIAX(self.fig.add_subplot(224), rturn)
        self._fig_shawn = False

    def close(self):
        plt.close(self.fig)

    def set_vision_lim(self, *lim):
        self.iax_vision.set_lim(*lim)

    def set_return_lim(self, *lim):
        self.iax_return.set_lim(*lim)

    def set_return_range(self, *rnge):
        self.iax_return.set_range(*rnge)

    def update(self, vision, target, pred, index):
        self.iax_vision(vision)
        self.iax_tactile_map(np.array([target, pred]))
        self.iax_return(index)
        if self._fig_shawn:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        else:
            self.fig.show()
            self._fig_shawn = True


class DatabaseDisplay:
    def __init__(self, path):
        self.dataset = database.get_dataset(path, vision=True, positions=True, speeds=True, tactile_map=True)
        self.iterator = self.dataset.make_initializable_iterator()
        self.next = self.iterator.get_next()
        self.initilalizer = self.iterator.initializer

    def __call__(self, t=None, n=None):
        with tf.Session() as sess:
            stop = False
            start_time = time.time()
            win = VisionJointsSkinWindow()
            sess.run(self.initilalizer)
            try:
                while not stop:
                    ret = sess.run(self.next)
                    win.update(ret["vision"], ret["positions"], ret["speeds"], ret["tactile_map"])
                    n = n - 1 if n is not None else None
                    elapsed_time = time.time() - start_time if t is not None else None
                    stop = (n is not None and n <= 0) or (elapsed_time is not None and elapsed_time > t)
            except tf.errors.OutOfRangeError:
                pass
        win.close()


class JointAgentWindow:
    def __init__(self, discount_factor, return_lookback=100):
        self.fig = plt.figure()
        self.iax_vision = VisionIAX(self.fig.add_subplot(321))
        self.iax_return = OnlineReturnIAX(self.fig.add_subplot(322), discount_factor, return_lookback)
        conf = [
            (325, "Arm1_to_Arm2_Left"),
            (326, "Arm1_to_Arm2_Right"),
            (323, "Ground_to_Arm1_Left"),
            (324, "Ground_to_Arm1_Right")]
        self.iax_joints = [DiscreteJointPositionIAX(self.fig.add_subplot(num), jname) for num, jname in conf]
        self._fig_shawn = False

    def close(self):
        plt.close(self.fig)

    def set_vision_lim(self, *lim):
        self.iax_vision.set_lim(*lim)

    def set_return_lim(self, *lim):
        self.iax_return.set_lim(*lim)

    def set_return_range(self, *rnge):
        self.iax_return.set_range(*rnge)

    def set_joint_lim(self, *lim):
        for iax in self.iax_joints:
            iax.set_lim(*lim)

    def __call__(self, vision, current_positions, target_positions, predicted_positions,
                 next_positions, current_reward, predicted_return):
        self.iax_vision(vision)
        self.iax_return(current_reward, predicted_return)
        for predicted_position, next_position, current_position, target_position, iax in \
                zip(predicted_positions, next_positions, current_positions, target_positions, self.iax_joints):
            iax(predicted_position, next_position, current_position, target_position)
        if self._fig_shawn:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        else:
            self.fig.show()
            self._fig_shawn = True


class SimpleJointAgentWindow:
    def __init__(self):
        self.fig = plt.figure()
        self.iax_vision = VisionIAX(self.fig.add_subplot(321))
        self.iax_tactile = TactileIAX(self.fig.add_subplot(322))
        conf = [
            (325, "Arm1_to_Arm2_Left"),
            (326, "Arm1_to_Arm2_Right"),
            (323, "Ground_to_Arm1_Left"),
            (324, "Ground_to_Arm1_Right")]
        self.iax_joints = [DiscreteJointPositionIAX(self.fig.add_subplot(num), jname) for num, jname in conf]
        self._fig_shawn = False

    def close(self):
        plt.close(self.fig)

    def set_vision_lim(self, *lim):
        self.iax_vision.set_lim(*lim)

    def set_return_lim(self, *lim):
        self.iax_return.set_lim(*lim)

    def set_return_range(self, *rnge):
        self.iax_return.set_range(*rnge)

    def set_joint_lim(self, *lim):
        for iax in self.iax_joints:
            iax.set_lim(*lim)

    def __call__(self, vision, positions, tactile):
        self.iax_vision(vision)
        self.iax_tactile(tactile)
        for position, iax in zip(positions, self.iax_joints):
            iax(position)
        if self._fig_shawn:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        else:
            self.fig.show()
            self._fig_shawn = True


class SkinAgentWindow:
    def __init__(self, discount_factor, return_lookback=100):
        self.fig = plt.figure()
        self.iax_vision = VisionIAX(self.fig.add_subplot(221))
        self.iax_return = OnlineReturnIAX(self.fig.add_subplot(222), discount_factor, return_lookback)
        self.iax_tactile_true = TactileIAX(self.fig.add_subplot(223))
        self.iax_tactile_target = TactileIAX(self.fig.add_subplot(224))
        self.annotation = self.fig.text(0.5, 0.3, "None")
        self._fig_shawn = False

    def close(self):
        plt.close(self.fig)

    def set_vision_lim(self, *lim):
        self.iax_vision.set_lim(*lim)

    def set_return_lim(self, *lim):
        self.iax_return.set_lim(*lim)

    def set_return_range(self, *rnge):
        self.iax_return.set_range(*rnge)

    def set_tactile_lim(self, *lim):
        self.iax_tactile.set_lim(*lim)

    def __call__(self, vision, tactile_true, tactile_target, current_reward, predicted_return):
        self.iax_vision(vision)
        self.iax_return(current_reward, predicted_return)
        self.iax_tactile_true(tactile_true)
        self.iax_tactile_target(tactile_target)
        self.annotation.set_text("{:.2f}".format(np.sum(np.abs(tactile_true - tactile_target))))
        if self._fig_shawn:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        else:
            self.fig.show()
            self._fig_shawn = True
