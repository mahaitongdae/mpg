#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: ampc.py
# =====================================

import logging

import gym
import numpy as np
from gym.envs.user_defined.toyota_env.dynamics_and_models import EnvironmentModel

from preprocessor import Preprocessor
from utils.misc import TimerStat

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AMPCLearner(object):
    import tensorflow as tf

    def __init__(self, policy_cls, args):
        self.args = args
        self.env = gym.make(self.args.env_id,
                            training_task=self.args.training_task,
                            num_future_data=self.args.num_future_data)
        obs_space, act_space = self.env.observation_space, self.env.action_space
        self.env.close()
        self.policy_with_value = policy_cls(obs_space, act_space, self.args)
        self.batch_data = {}
        self.all_data = {}
        self.M = self.args.M
        self.num_rollout_list_for_policy_update = self.args.num_rollout_list_for_policy_update

        self.model = EnvironmentModel(task=self.args.training_task,
                                      num_future_data=self.args.num_future_data)
        self.preprocessor = Preprocessor(obs_space, self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                         self.args.obs_scale_factor, self.args.reward_scale_factor,
                                         gamma=self.args.gamma)
        self.policy_gradient_timer = TimerStat()
        self.stats = {}
        self.info_for_buffer = {}

    def get_stats(self):
        return self.stats

    def get_info_for_buffer(self):
        return self.info_for_buffer

    def get_batch_data(self, batch_data, rb, indexes):
        self.batch_data = {'batch_obs': batch_data[0].astype(np.float32),
                           'batch_actions': batch_data[1].astype(np.float32),
                           'batch_rewards': batch_data[2].astype(np.float32),
                           'batch_obs_tp1': batch_data[3].astype(np.float32),
                           'batch_dones': batch_data[4].astype(np.float32),
                           }

        # print(self.batch_data['batch_obs'].shape)  # batch_size * obs_dim
        # print(self.batch_data['batch_actions'].shape)  # batch_size * act_dim
        # print(self.batch_data['batch_advs'].shape)  # batch_size,
        # print(self.batch_data['batch_tdlambda_returns'].shape)  # batch_size,

    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def punish_factor_schedule(self, ite):
        init_pf = self.args.init_punish_factor
        interval = self.args.pf_enlarge_interval
        amplifier = self.args.pf_amplifier
        pf = init_pf * self.tf.pow(amplifier, self.tf.cast(ite//interval, self.tf.float32))
        return pf

    def model_rollout_for_policy_update(self, start_obses, ite):
        start_obses = self.tf.tile(start_obses, [self.M, 1])
        self.model.reset(start_obses, self.args.training_task)
        rewards_sum = self.tf.zeros((start_obses.shape[0],))
        punish_terms_sum = self.tf.zeros((start_obses.shape[0],))
        obses = start_obses
        pf = self.punish_factor_schedule(ite)

        for _ in range(self.num_rollout_list_for_policy_update[0]):
            processed_obses = self.preprocessor.tf_process_obses(obses)
            actions, _ = self.policy_with_value.compute_action(processed_obses)
            obses, rewards, punish_terms = self.model.rollout_out(actions)
            rewards_sum += self.preprocessor.tf_process_rewards(rewards)
            punish_terms_sum += punish_terms

        obj_loss = -self.tf.reduce_mean(rewards_sum)
        punish_term = self.tf.reduce_mean(punish_terms_sum)
        punish_loss = self.tf.stop_gradient(pf) * punish_term
        total_loss = obj_loss + punish_loss

        return obj_loss, punish_term, punish_loss, total_loss, pf

    @tf.function
    def policy_forward_and_backward(self, mb_obs, ite):
        with self.tf.GradientTape() as tape:
            obj_loss, punish_term, punish_loss, total_loss, pf = self.model_rollout_for_policy_update(mb_obs, ite)

        with self.tf.name_scope('policy_gradient') as scope:
            policy_gradient = tape.gradient(total_loss, self.policy_with_value.policy.trainable_weights)
            return policy_gradient, obj_loss, punish_term, punish_loss, total_loss, pf

    def export_graph(self, writer):
        mb_obs = self.batch_data['batch_obs']
        self.tf.summary.trace_on(graph=True, profiler=False)
        self.policy_forward_and_backward(mb_obs, self.tf.convert_to_tensor(0, self.tf.int32))
        with writer.as_default():
            self.tf.summary.trace_export(name="policy_forward_and_backward", step=0)

    def compute_gradient(self, samples, rb, indexs, iteration):
        self.get_batch_data(samples, rb, indexs)
        mb_obs = self.batch_data['batch_obs']
        iteration = self.tf.convert_to_tensor(iteration, self.tf.int32)

        with self.policy_gradient_timer:
            policy_gradient, obj_loss, punish_term, punish_loss, total_loss, pf =\
                self.policy_forward_and_backward(mb_obs, iteration)
            policy_gradient, policy_gradient_norm = self.tf.clip_by_global_norm(policy_gradient,
                                                                                self.args.gradient_clip_norm)

        self.stats.update(dict(
            iteration=iteration,
            pg_time=self.policy_gradient_timer.mean,
            obj_loss=obj_loss.numpy(),
            punish_term=punish_term.numpy(),
            punish_loss=punish_loss.numpy(),
            total_loss=total_loss.numpy(),
            punish_factor=pf.numpy(),
            policy_gradient_norm=policy_gradient_norm.numpy(),
        ))

        gradient_tensor = policy_gradient
        return list(map(lambda x: x.numpy(), gradient_tensor))


if __name__ == '__main__':
    pass
