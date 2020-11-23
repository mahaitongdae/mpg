#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: worker.py
# =====================================

import logging
import os
from collections import deque

import gym
import numpy as np

from preprocessor import Preprocessor
from utils.misc import TimerStat, safemean
from utils.monitor import Monitor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class OnPolicyWorker(object):
    """
    Act as both actor and learner
    """
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')

    def __init__(self, policy_cls, learner_cls, env_id, args, worker_id):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self.worker_id = worker_id
        self.args = args
        env = gym.make(env_id)
        self.env = Monitor(env)
        obs_space, act_space = self.env.observation_space, self.env.action_space
        self.policy_with_value = policy_cls(obs_space, act_space, self.args)
        self.learner = learner_cls(self.policy_with_value, self.args)
        self.sample_batch_size = self.args.sample_batch_size
        self.obs = self.env.reset()
        self.done = False
        self.preprocessor = Preprocessor(obs_space, self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                         self.args.obs_scale, self.args.reward_scale, self.args.reward_shift,
                                         gamma=self.args.gamma)
        self.log_dir = self.args.log_dir

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.stats = {}
        self.sampling_timer = TimerStat()
        self.processing_timer = TimerStat()
        self.epinfobuf = deque(maxlen=100)
        logger.info('Worker initialized')

    def get_stats(self):
        return self.stats

    def save_weights(self, save_dir, iteration):
        self.policy_with_value.save_weights(save_dir, iteration)

    def load_weights(self, load_dir, iteration):
        self.policy_with_value.load_weights(load_dir, iteration)

    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def apply_grads_sepe(self, grads):
        self.policy_with_value.apply_grads_sepe(grads)

    def apply_grads_all(self, grads, lr):
        lr = self.tf.constant(lr, dtype=self.tf.float32)
        self.policy_with_value.apply_grads_all(grads, lr)

    def get_ppc_params(self):
        return self.preprocessor.get_params()

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def save_ppc_params(self, save_dir):
        self.preprocessor.save_params(save_dir)

    def load_ppc_params(self, load_dir):
        self.preprocessor.load_params(load_dir)

    def sample_and_process(self):
        with self.sampling_timer:
            mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_logps = [], [], [], [], [], []

            epinfos = []
            batch_data = []
            for _ in range(self.sample_batch_size):
                processed_obs = self.preprocessor.process_obs(self.obs)
                processed_obs_tensor = self.tf.constant(processed_obs[np.newaxis, :])
                action, logp = self.policy_with_value.compute_action(processed_obs_tensor)
                value = self.policy_with_value.compute_vf(processed_obs_tensor)
                action, logp, value = action.numpy()[0], logp.numpy()[0], value.numpy()[0]

                mb_obs.append(processed_obs.copy())
                mb_actions.append(action)
                mb_values.append(value)
                mb_logps.append(logp)
                mb_dones.append(self.done)

                obs_tp1, reward, self.done, info = self.env.step(action)
                processed_rew = self.preprocessor.process_rew(reward, self.done)
                self.obs = self.env.reset() if self.done else obs_tp1.copy()
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)
                mb_rewards.append(processed_rew)

                # batch_data.append((processed_obs.copy(), action, processed_rew, obs_tp1, self.done, logp))
            mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
            mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
            mb_actions = np.asarray(mb_actions)
            mb_values = np.asarray(mb_values, dtype=np.float32)
            mb_logps = np.asarray(mb_logps, dtype=np.float32)
            mb_dones = np.asarray(mb_dones, dtype=np.bool)
            processed_obs = self.preprocessor.process_obs(self.obs)
            processed_obs_tensor = self.tf.constant(processed_obs[np.newaxis, :])
            last_values = self.policy_with_value.compute_vf(processed_obs_tensor).numpy()[0]

            mb_advs = np.zeros_like(mb_rewards)
            lastgaelam = 0
            for t in reversed(range(self.sample_batch_size-1)):
                # if t == self.sample_batch_size - 1:
                #     nextnonterminal = 1.0 - self.done
                #     nextvalues = last_values
                # else:
                nextnonterminal = 1.0 - mb_dones[t + 1]
                nextvalues = mb_values[t + 1]
                delta = mb_rewards[t] + self.args.gamma * nextvalues * nextnonterminal - mb_values[t]
                mb_advs[t] = lastgaelam = delta + self.args.gamma * self.args.lam * nextnonterminal * lastgaelam
            mb_returns = mb_advs + mb_values
            batch_data = dict(batch_obs=mb_obs,
                              batch_actions=mb_actions,
                              batch_rewards=mb_rewards,
                              batch_dones=mb_dones,
                              batch_logps=mb_logps,
                              batch_advs=mb_advs,
                              batch_tdlambda_returns=mb_returns,
                              batch_values=mb_values)
        with self.processing_timer:
            self.learner.get_batch_data(batch_data)
        ev = 1. - np.var(batch_data['batch_tdlambda_returns']-batch_data['batch_values'])/np.var(batch_data['batch_tdlambda_returns'])

        print(ev)
        self.epinfobuf.extend(epinfos)
        self.stats.update(explained_variance=ev,
                          eprewmean=safemean([epinfo['r'] for epinfo in self.epinfobuf]),
                          eplenmean=safemean([epinfo['l'] for epinfo in self.epinfobuf]))
        self.stats.update(dict(worker_sampling_time=self.sampling_timer.mean,
                               worker_processing_time=self.processing_timer.mean))
        if self.args.reward_preprocess_type == 'normalize':
            self.stats.update(dict(ret_rms_var=self.preprocessor.ret_rms.var,
                                   ret_rms_mean=self.preprocessor.ret_rms.mean))

    def compute_gradient_over_ith_minibatch(self, i):
        grad = self.learner.compute_gradient_over_ith_minibatch(i)
        learner_stats = self.learner.get_stats()
        self.stats.update(learner_stats)
        return grad


class PPOWorker(object):
    """
    Act as both actor and learner
    """
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')

    def __init__(self, policy_cls, learner_cls, env_id, args, worker_id):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self.worker_id = worker_id
        self.args = args
        env = gym.make(env_id)
        self.env = Monitor(env)
        obs_space, act_space = self.env.observation_space, self.env.action_space
        self.policy_with_value = policy_cls(obs_space, act_space, self.args)
        self.sample_batch_size = self.args.sample_batch_size
        self.obs = self.env.reset()
        self.done = False
        self.preprocessor = Preprocessor(obs_space, self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                         self.args.obs_scale, self.args.reward_scale, self.args.reward_shift,
                                         gamma=self.args.gamma)
        self.log_dir = self.args.log_dir

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.stats = {}
        self.sampling_timer = TimerStat()
        self.processing_timer = TimerStat()
        self.epinfobuf = deque(maxlen=100)
        self.batch_data = None
        logger.info('Worker initialized')

    def get_stats(self):
        return self.stats

    def save_weights(self, save_dir, iteration):
        self.policy_with_value.save_weights(save_dir, iteration)

    def load_weights(self, load_dir, iteration):
        self.policy_with_value.load_weights(load_dir, iteration)

    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def apply_grads_sepe(self, grads):
        self.policy_with_value.apply_grads_sepe(grads)

    def apply_grads_all(self, grads, lr):
        lr = self.tf.constant(lr, dtype=self.tf.float32)
        self.policy_with_value.apply_grads_all(grads, lr)

    def get_ppc_params(self):
        return self.preprocessor.get_params()

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def save_ppc_params(self, save_dir):
        self.preprocessor.save_params(save_dir)

    def load_ppc_params(self, load_dir):
        self.preprocessor.load_params(load_dir)

    def sample_and_process(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_logps = [], [], [], [], [], []
        epinfos = []
        for _ in range(self.sample_batch_size):
            processed_obs = self.preprocessor.process_obs(self.obs)
            processed_obs_tensor = self.tf.constant(processed_obs[np.newaxis, :])
            action, logp = self.policy_with_value.compute_action(processed_obs_tensor)
            value = self.policy_with_value.compute_vf(processed_obs_tensor)
            action, logp, value = action.numpy()[0], logp.numpy()[0], value.numpy()[0]

            mb_obs.append(processed_obs.copy())
            mb_actions.append(action)
            mb_values.append(value)
            mb_logps.append(logp)
            mb_dones.append(self.done)

            obs_tp1, reward, self.done, info = self.env.step(action)
            processed_rew = self.preprocessor.process_rew(reward, self.done)
            self.obs = self.env.reset() if self.done else obs_tp1.copy()
            maybeepinfo = info.get('episode')
            if maybeepinfo:
                epinfos.append(maybeepinfo)
            mb_rewards.append(processed_rew)

        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_logps = np.asarray(mb_logps, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        processed_obs = self.preprocessor.process_obs(self.obs)
        processed_obs_tensor = self.tf.constant(processed_obs[np.newaxis, :])
        last_values = self.policy_with_value.compute_vf(processed_obs_tensor).numpy()[0]

        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.sample_batch_size-1)):
            if t == self.sample_batch_size - 1:
                nextnonterminal = 1.0 - self.done
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t + 1]
                nextvalues = mb_values[t + 1]
            delta = mb_rewards[t] + self.args.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.args.gamma * self.args.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        self.batch_data = dict(batch_obs=mb_obs,
                               batch_actions=mb_actions,
                               batch_rewards=mb_rewards,
                               batch_dones=mb_dones,
                               batch_logps=mb_logps,
                               batch_advs=mb_advs,
                               batch_tdlambda_returns=mb_returns,
                               batch_values=mb_values)
        ev = 1. - np.var(self.batch_data['batch_tdlambda_returns']-self.batch_data['batch_values'])/np.var(self.batch_data['batch_tdlambda_returns'])
        self.epinfobuf.extend(epinfos)
        self.stats.update(explained_variance=ev,
                          eprewmean=safemean([epinfo['r'] for epinfo in self.epinfobuf]),
                          eplenmean=safemean([epinfo['l'] for epinfo in self.epinfobuf]))
        if self.args.reward_preprocess_type == 'normalize':
            self.stats.update(dict(ret_rms_var=self.preprocessor.ret_rms.var,
                                   ret_rms_mean=self.preprocessor.ret_rms.mean))

    @tf.function
    def get_grads(self, mb_obs, mb_actions, mb_logps, mb_advs, target, mb_oldvs):
        mb_advs = (mb_advs - self.tf.reduce_mean(mb_advs)) / (self.tf.keras.backend.std(mb_advs) + 1e-8)
        with self.tf.GradientTape() as tape:
            v_pred = self.policy_with_value.compute_vf(mb_obs)
            vpredclipped = mb_oldvs + self.tf.clip_by_value(v_pred - mb_oldvs,
                                                            -self.args.ppo_loss_clip,
                                                            self.args.ppo_loss_clip)
            v_loss1 = self.tf.square(v_pred - target)
            v_loss2 = self.tf.square(vpredclipped - target)
            v_loss = .5 * self.tf.reduce_mean(self.tf.maximum(v_loss1, v_loss2))

            current_logp = self.policy_with_value.compute_logps(mb_obs, mb_actions)
            ratio = self.tf.exp(current_logp - mb_logps)
            pg_loss1 = ratio * mb_advs
            pg_loss2 = mb_advs * self.tf.clip_by_value(ratio, 1 - self.args.ppo_loss_clip, 1 + self.args.ppo_loss_clip)
            pg_loss = -self.tf.reduce_mean(self.tf.minimum(pg_loss1, pg_loss2))

            policy_entropy = self.policy_with_value.compute_entropy(mb_obs)
            ent_bonus = self.args.ent_coef * policy_entropy

            value_mean = self.tf.reduce_mean(v_pred)
            approxkl = .5 * self.tf.reduce_mean(self.tf.square(current_logp - mb_logps))
            clipfrac = self.tf.reduce_mean(self.tf.cast(
                self.tf.greater(self.tf.abs(ratio - 1.0), self.args.ppo_loss_clip), self.tf.float32))

            total_loss = v_loss + pg_loss - ent_bonus

        grads = tape.gradient(total_loss, self.policy_with_value.trainable_variables)
        grad, grad_norm = self.tf.clip_by_global_norm(grads, self.args.gradient_clip_norm)
        return grad, grad_norm, pg_loss, ent_bonus, policy_entropy, clipfrac, v_loss, value_mean, approxkl

    def compute_gradient_over_ith_minibatch(self, i):  # compute gradient of the i-th mini-batch
        if i == 0:
            self.permutation = np.arange(self.args.sample_batch_size)
            np.random.shuffle(self.permutation)
        start_idx, end_idx = i * self.args.mini_batch_size, (i + 1) * self.args.mini_batch_size
        mbinds = self.permutation[start_idx:end_idx]
        mb_obs = self.tf.constant(self.batch_data['batch_obs'][mbinds])
        mb_advs = self.tf.constant(self.batch_data['batch_advs'][mbinds])
        mb_tdlambda_returns = self.tf.constant(self.batch_data['batch_tdlambda_returns'][mbinds])
        mb_actions = self.tf.constant(self.batch_data['batch_actions'][mbinds])
        mb_logps = self.tf.constant(self.batch_data['batch_logps'][mbinds])
        mb_oldvs = self.tf.constant(self.batch_data['batch_values'][mbinds])

        grad, grad_norm, pg_loss, ent_bonus, policy_entropy, clipfrac, v_loss, value_mean, approxkl = \
            self.get_grads(mb_obs, mb_actions, mb_logps, mb_advs, mb_tdlambda_returns, mb_oldvs)

        self.stats = dict(
            v_loss=v_loss.numpy(),
            policy_loss=pg_loss.numpy(),
            ent_bonus=ent_bonus.numpy(),
            policy_entropy=policy_entropy.numpy(),
            value_mean=value_mean.numpy(),
            target_mean=np.mean(mb_tdlambda_returns),
            grad_norm=grad_norm.numpy(),
            clipfrac=clipfrac.numpy(),
            approxkl=approxkl.numpy()
        )

        return grad


def debug_worker():
    from train_script import built_PPO_parser
    from policy import PolicyWithValue
    from learners.ppo import PPOLearner
    env_id = 'Pendulum-v0'
    worker_id = 0
    args = built_PPO_parser()
    worker = OnPolicyWorker(PolicyWithValue, PPOLearner, env_id, args, worker_id)
    for _ in range(10):
        worker.sample_and_process()


if __name__ == '__main__':
    debug_worker()

