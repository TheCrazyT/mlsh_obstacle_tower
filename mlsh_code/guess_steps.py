from rl_algs.common.mpi_running_mean_std import RunningMeanStd
import rl_algs.common.tf_util as U
import tensorflow as tf
import numpy as np
import gym
from rl_algs.common.distributions import CategoricalPdType
from rl_algs.common.mpi_running_mean_std import RunningMeanStd


class GuessStepsPolicy(object):
    def __init__(self, name, ob, hid_size, num_hid_layers, gaussian_fixed_var=True):
        self.hid_size = hid_size
        self.num_hid_layers = num_hid_layers
        self.gaussian_fixed_var = gaussian_fixed_var

        with tf.variable_scope(name):
            self.scope = tf.get_variable_scope().name
            with tf.variable_scope("obfilter"):
                if(len(ob.shape)==2):
                    self.ob_rms = RunningMeanStd(shape=(ob.get_shape()[1],))
                elif(len(ob.shape)==3):
                    self.ob_rms = RunningMeanStd(shape=(ob.get_shape()[1] * ob.get_shape()[2]))
                elif(len(ob.shape)==4):
                    self.ob_rms = RunningMeanStd(shape=(ob.get_shape()[1] * ob.get_shape()[2] * ob.get_shape()[3]))
            #obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            obz = ob

            # value function
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(U.dense(last_out, hid_size, "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
            self.vpred = tf.clip_by_value(U.sum(U.dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:,0]),0.0,1000.0)

        # sample actions
        self._act = U.function([ob], [self.vpred])

    def act(self, ob):
        vpred1 =  self._act(ob[None])
        return vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def reset(self):
        with tf.variable_scope(self.scope, reuse=True):
            varlist = self.get_trainable_variables()
            initializer = tf.variables_initializer(varlist)
            U.get_session().run(initializer)