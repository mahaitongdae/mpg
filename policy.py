#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/11/09
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: policy.py
# =====================================

import tensorflow as tf
from gym import spaces

from model import MLPNet, MLPNetDSAC, PPONet
from tensorflow.keras.optimizers.schedules import PolynomialDecay, PiecewiseConstantDecay

NAME2MODELCLS = dict([('MLP', MLPNet), ('DSAC', MLPNetDSAC), ('PPO', PPONet)])


class PolicyWithValue(tf.Module):
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    tfb = tfp.bijectors
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, obs_space, act_space, args):
        super().__init__()
        self.args = args
        assert isinstance(obs_space, spaces.Box)
        assert isinstance(act_space, spaces.Box)
        obs_dim = obs_space.shape[0] if args.obs_dim is None else self.args.obs_dim
        act_dim = act_space.shape[0] if args.act_dim is None else self.args.act_dim
        n_hiddens, n_units, hidden_activation = self.args.num_hidden_layers, self.args.num_hidden_units, self.args.hidden_activation
        value_model_cls, policy_model_cls = NAME2MODELCLS[self.args.value_model_cls], \
                                            NAME2MODELCLS[self.args.policy_model_cls]
        self.policy = policy_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, act_dim * 2, name='policy',
                                       output_activation=self.args.policy_out_activation)
        self.value = value_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, 1, name='value')
        self.models = (self.policy, self.value,)
        policy_lr_schedule = self.tf.keras.optimizers.schedules.PolynomialDecay(*self.args.policy_lr_schedule)
        value_lr_schedule = self.tf.keras.optimizers.schedules.PolynomialDecay(*self.args.value_lr_schedule)
        self.policy_optimizer = self.tf.keras.optimizers.Adam(policy_lr_schedule)
        self.value_optimizer = self.tf.keras.optimizers.Adam(value_lr_schedule)
        self.optimizers = (self.policy_optimizer, self.value_optimizer,)

    def save_weights(self, save_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.save(save_dir + '/ckpt_ite' + str(iteration))

    def load_weights(self, load_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.restore(load_dir + '/ckpt_ite' + str(iteration) + '-1')

    def get_weights(self):
        return [model.get_weights() for model in self.models]

    def set_weights(self, weights):
        for i, weight in enumerate(weights):
            self.models[i].set_weights(weight)

    @tf.function
    def apply_grads_sepe(self, grads):
        value_weights_len = len(self.value.trainable_weights)
        value_grad, policy_grad = grads[:value_weights_len], grads[value_weights_len:]
        self.value_optimizer.apply_gradients(zip(value_grad, self.value.trainable_weights))
        self.policy_optimizer.apply_gradients(zip(policy_grad, self.policy.trainable_weights))

    @tf.function
    def apply_grads_all(self, grads):
        self.policy_optimizer.apply_gradients(zip(grads, self.trainable_variables))

    def _logits2dist(self, logits):
        mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        act_dist = self.tfd.MultivariateNormalDiag(mean, self.tf.exp(log_std))
        if self.args.action_range is not None:
            act_dist = (
                self.tfp.distributions.TransformedDistribution(
                    distribution=act_dist,
                    bijector=self.tfb.Chain(
                        [self.tfb.Affine(scale_identity_multiplier=self.args.action_range),
                         self.tfb.Tanh()])
                ))
        return act_dist

    @tf.function
    def compute_action(self, obs):
        with self.tf.name_scope('compute_action') as scope:
            logits = self.policy(obs)
            act_dist = self._logits2dist(logits)
            actions = act_dist.sample()
            logps = act_dist.log_prob(actions)
            return actions, logps

    @tf.function
    def compute_logps(self, obs, actions):
        with self.tf.name_scope('compute_logps') as scope:
            logits = self.policy(obs)
            act_dist = self._logits2dist(logits)
            actions = self.tf.clip_by_value(actions, -self.args.action_range+0.01, self.args.action_range-0.01)
            return act_dist.log_prob(actions)

    @tf.function
    def compute_entropy(self, obs):
        with self.tf.name_scope('compute_entropy') as scope:
            logits = self.policy(obs)
            act_dist = self._logits2dist(logits)
            try:
                entropy = self.tf.reduce_mean(act_dist.entropy())
            except NotImplementedError:
                actions = act_dist.sample()
                logps = act_dist.log_prob(actions)
                entropy = -self.tf.reduce_mean(logps)
            finally:
                return entropy

    @tf.function
    def compute_kl(self, obs, other_out):  # KL(other||ego)
        with self.tf.name_scope('compute_entropy') as scope:
            logits = self.policy(obs)
            act_dist = self._logits2dist(logits)
            other_act_dist = self._logits2dist(self.tf.stop_gradient(other_out))
            try:
                kl = self.tf.reduce_mean(other_act_dist.kl_divergence(act_dist))
            except NotImplementedError:
                other_actions = other_act_dist.sample()
                other_logps = other_act_dist.log_prob(other_actions)
                logps = self.compute_logps(obs, other_actions)
                kl = self.tf.reduce_mean(other_logps - logps)
            finally:
                return kl

    @tf.function
    def compute_mode(self, obs):
        logits = self.policy(obs)
        mean, _ = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        return self.args.action_range * self.tf.tanh(mean) if self.args.action_range is not None else mean

    @tf.function
    def compute_vf(self, obs):
        with self.tf.name_scope('compute_value') as scope:
            return tf.squeeze(self.value(obs), axis=1)

class AttnPolicy4Lagrange(tf.Module):
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    tfb = tfp.bijectors
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, args):
        super().__init__()
        self.args = args

        obs_dim, act_dim = self.args.obs_dim, self.args.act_dim
        mu_dim = self.args.con_dim
        veh_dim = self.args.veh_dim
        veh_num = self.args.veh_num
        ego_dim = self.args.ego_dim
        tracking_dim = self.args.tracking_dim

        d_model = self.args.d_model
        num_attn_layers = self.args.num_attn_layers
        d_ff = self.args.d_ff
        num_heads = self.args.num_heads
        dropout = self.args.drop_rate
        max_len = self.args.max_veh_num

        assert tracking_dim + ego_dim + veh_dim*veh_num == obs_dim
        assert 4 + veh_num * 4 == mu_dim

        backbone_cls = NAME2MODELCLS[self.args.backbone_cls]

        # Attention backbone
        self.backbone = backbone_cls(ego_dim, obs_dim-tracking_dim-ego_dim, veh_num, tracking_dim,
                                     num_attn_layers, d_model, d_ff, num_heads, dropout,
                                     max_len, name='backbone')
        mu_value_lr_schedule = PolynomialDecay(*self.args.mu_lr_schedule)
        self.mu_optimizer = self.tf.optimizers.Adam(mu_value_lr_schedule, name='mu_adam_opt')

        self.policy = Sequential([tf.keras.layers.InputLayer(input_shape=(d_model,)),
                                  Dense(d_model, activation=self.args.policy_out_activation,
                                        kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.)),
                                        dtype=tf.float32),
                                  Dense(act_dim * 2, activation=self.args.policy_out_activation,
                                        kernel_initializer=tf.keras.initializers.Orthogonal(1.),
                                        bias_initializer = tf.keras.initializers.Constant(0.),
                                        dtype = tf.float32),])
        policy_lr_schedule = PolynomialDecay(*self.args.policy_lr_schedule)
        self.policy_optimizer = self.tf.keras.optimizers.Adam(policy_lr_schedule, name='adam_opt')

        # self.value = Sequential([tf.keras.Input(shape=(d_model,)),
        #                          Dense(1, activation='linear',
        #                                kernel_initializer=tf.keras.initializers.Orthogonal(1.),
        #                                bias_initializer=tf.keras.initializers.Constant(0.),
        #                                dtype=tf.float32),])
        # value_lr_schedule = PolynomialDecay(*self.args.value_lr_schedule)
        # self.value_optimizer = self.tf.keras.optimizers.Adam(value_lr_schedule, name='v_adam_opt')

        self.models = (self.backbone, self.policy)
        self.optimizers = (self.mu_optimizer, self.policy_optimizer)

    def save_weights(self, save_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.save(save_dir + '/ckpt_ite' + str(iteration))

    def load_weights(self, load_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.restore(load_dir + '/ckpt_ite' + str(iteration) + '-1')

    def get_weights(self):
        return [model.get_weights() for model in self.models]

    def set_weights(self, weights):
        for i, weight in enumerate(weights):
            self.models[i].set_weights(weight)

    @tf.function
    def apply_gradients(self, iteration, grads):
        policy_len = len(self.policy.trainable_weights)
        policy_grad, mu_grad = grads[:policy_len], grads[policy_len:]
        self.policy_optimizer.apply_gradients(zip(policy_grad, self.policy.trainable_weights))
        if iteration % self.args.mu_update_interval == 0:
            self.mu_optimizer.apply_gradients(zip(mu_grad, self.backbone.trainable_weights))

    @tf.function
    def compute_mode(self, obs):
        logits = self.policy(obs)
        mean, _ = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        return self.args.action_range * self.tf.tanh(mean) if self.args.action_range is not None else mean

    def _logits2dist(self, logits):
        mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        act_dist = self.tfd.MultivariateNormalDiag(mean, self.tf.exp(log_std))
        if self.args.action_range is not None:
            act_dist = (
                self.tfp.distributions.TransformedDistribution(
                    distribution=act_dist,
                    bijector=self.tfb.Chain(
                        [self.tfb.Affine(scale_identity_multiplier=self.args.action_range),
                         self.tfb.Tanh()])
                ))
        return act_dist

    @tf.function
    def compute_mu(self, obs, nonpadding_ind, training=True):
        def create_padding_mask(batch_size, seq_len, nonpadding_ind):
            nonpadding_ind = tf.cast(nonpadding_ind, dtype=tf.float32)
            nonpadding_ind = tf.concat([tf.ones((batch_size,1)), nonpadding_ind], axis=-1)
            nonpadding_ind = tf.reshape(nonpadding_ind, (batch_size, 1, -1))
            repaet_times = tf.constant([1, seq_len, 1], tf.int32)

            return tf.tile(nonpadding_ind, repaet_times)

        def create_mu_mask(batch_size, seq_len):
            mask = np.identity(seq_len, dtype=np.float32)
            mask[:, 0] = 1
            mask[0, :] = 1
            mask = mask[np.newaxis, :, :]
            return tf.convert_to_tensor(np.repeat(mask, repeats=batch_size, axis=0), dtype=tf.float32)

        with self.tf.name_scope('compute_mu') as scope:
            batch_size = (obs).shape[0]
            seq_len = self.args.veh_num+1
            x_ego = tf.expand_dims(obs[:, :self.args.ego_dim+self.args.tracking_dim], axis=1)
            x_vehs = tf.reshape(obs[:, self.args.ego_dim+self.args.tracking_dim:], (batch_size, -1, self.args.veh_dim))

            assert x_vehs.shape[1] == self.args.veh_num

            hidden, attn_weights = self.backbone(x_ego, x_vehs,
                                                 padding_mask=create_padding_mask(batch_size, seq_len, nonpadding_ind),
                                                 mu_mask=create_mu_mask(batch_size, seq_len),
                                                 training=training)
            mu_attn = attn_weights[:, :, 0, 1:]
            return hidden[:, 0, :], tf.cast(tf.exp(5*mu_attn)-1, dtype=tf.float32)

    @tf.function
    def compute_action(self, obs, nonpadding_ind, training=True):
        hidden, _ = self.compute_mu(obs, nonpadding_ind, training)
        hidden = tf.stop_gradient(hidden)
        with self.tf.name_scope('compute_action') as scope:
            logits = self.policy(hidden)
            if self.args.deterministic_policy:
                mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
                return self.args.action_range * self.tf.tanh(mean) if self.args.action_range is not None else mean, 0.
            else:
                act_dist = self._logits2dist(logits)
                actions = act_dist.sample()
                logps = act_dist.log_prob(actions)
                return actions, logps

    # @tf.function
    # def compute_v(self, hidden):
    #     with self.tf.name_scope('compute_v') as scope:
    #         return tf.squeeze(self.value(hidden), axis=1)


def test_logps():
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    tfb = tfp.bijectors

    import numpy as np
    mean, log_std = np.array([[10.]], np.float32), np.array([[-5.]], np.float32)
    std = np.exp(log_std)
    act_dist = tfd.MultivariateNormalDiag(mean, tf.exp(log_std))
    act_dist = (
            tfp.distributions.TransformedDistribution(
                distribution=act_dist,
                bijector=tfb.Chain(
                    [tfb.Affine(scale_identity_multiplier=1.),
                     tfb.Tanh()])))
    actions_sampled = act_dist.sample()
    actions = [[-0.99999]]
    logp_sampled = act_dist.log_prob(actions_sampled)
    logp = act_dist.log_prob(actions)
    print(std, actions_sampled, logp_sampled, actions, logp)

def test_logps2():
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    tfb = tfp.bijectors

    import numpy as np
    mean, log_std = np.array([[1000]], np.float32), np.array([[-5.]], np.float32)
    act_dist = tfd.Normal(mean, tf.exp(log_std))
    actions = [[1.-0.01]]#act_dist.sample()
    actions_norm = actions
    actions_sampled = tf.math.atanh(actions_norm)
    logp = act_dist.log_prob(actions_sampled) - tf.math.log(1. - tf.square(actions_norm))
    # 猜测nan的根源在inf-inf 解决办法就是不能要边界值
    logp = tf.reduce_sum(logp, axis=-1, keepdims=True)

    print(actions, logp)

def test_logps3():
    import torch
    from torch.distributions import Normal
    mean, log_std = torch.Tensor([[0.]]), torch.Tensor([[100000.]])
    act_dist = Normal(mean, log_std.exp())
    actions = torch.Tensor([[0.1]])#act_dist.sample()
    actions_norm = actions
    actions_sampled = torch.atanh(actions_norm)
    print(act_dist.log_prob(actions_sampled))
    logp = act_dist.log_prob(actions_sampled) - torch.log(1. - actions_norm.pow(2))
    logp = logp.sum(dim=-1, keepdim=True)

    print(actions, logp)




def testMultivariateNormalDiag():
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    import numpy as np
    mean, log_std = np.array([[0.1]]), np.array([[0.1]])

    dist = tfd.MultivariateNormalDiag(mean, tf.exp(log_std))
    print(dist.sample())


if __name__ == "__main__":
    test_logps()
    test_logps2()