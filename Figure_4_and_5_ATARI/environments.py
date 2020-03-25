import collections
import tensorflow as tf
from tensorflow import nest

import aleop
import cv2


StepOutputInfo = collections.namedtuple('StepOutputInfo',
                                        'episode_return episode_step')
StepOutput = collections.namedtuple('StepOutput',
                                    'reward info done observation')


def create_environment(level_name, num_action_repeats, queue=None, max_episode_length=-1, should_reset=0):
    def py_func(_img):
        if queue is not None:
            queue.put(_img)
        _img = cv2.cvtColor(_img, cv2.COLOR_RGB2GRAY)
        _img = cv2.resize(_img, (84, 84), interpolation=cv2.INTER_AREA)[:, :, None]
        return _img

    def preprocess(img):
        t = tf.py_func(py_func, [img], tf.uint8)
        t.set_shape((84, 84, 1))
        return t

    level_name = level_name[level_name.index('atari/') + 6:]

    class Env:
        def initial(self):
            obs = tf.zeros((210, 160, 3), tf.uint8)
            obs = preprocess(obs)
            initial_info = StepOutputInfo(episode_return=tf.constant(0.),
                                          episode_step=tf.constant(0, tf.int64))
            return StepOutput(reward=tf.zeros(()),
                              info=initial_info,
                              done=tf.zeros((), dtype=tf.bool),
                              observation=[obs]), initial_info

        def step(self, action, state):
            state = nest.map_structure(lambda x: tf.where(should_reset > 0, tf.zeros_like(x), x), state)
            reward, done, obs = aleop.ale(action, should_reset, max_episode_length, level_name,
                                          frameskip_min=num_action_repeats,
                                          frameskip_max=num_action_repeats)
            reward.set_shape(())
            reward = tf.clip_by_value(reward, -1., 1.)
            done.set_shape(())
            obs.set_shape((210, 160, 3))
            obs = preprocess(obs)
            state_episode_return = tf.where(done, tf.zeros(()), state.episode_return + reward)
            state_episode_step = tf.where(done, tf.zeros((), tf.int64), state.episode_step + 1)
            state_updated_info = StepOutputInfo(episode_return=state_episode_return,
                                                episode_step=state_episode_step)
            updated_info = StepOutputInfo(episode_return=state.episode_return + reward,
                                          episode_step=state.episode_step + 1)
            return StepOutput(reward=reward,
                              info=updated_info,
                              done=done,
                              observation=[obs]), state_updated_info

    return Env()
