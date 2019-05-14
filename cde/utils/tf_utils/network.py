#
# Code from rllab https://github.com/rll/rllab/tree/master/sandbox
#

import cde.utils.tf_utils.layers as L
import tensorflow as tf
import numpy as np
import itertools
from cde.utils.serializable import Serializable
from cde.utils.tf_utils.parameterized import Parameterized
from cde.utils.tf_utils.layers_powered import LayersPowered


class MLP(LayersPowered, Serializable):
    def __init__(self, name, output_dim, hidden_sizes, hidden_nonlinearity,
                 output_nonlinearity, hidden_W_init=L.XavierUniformInitializer(), hidden_b_init=tf.zeros_initializer(),
                 output_W_init=L.XavierUniformInitializer(), output_b_init=tf.zeros_initializer(),
                 input_var=None, input_layer=None, input_shape=None, batch_normalization=False, weight_normalization=False,
                 dropout_ph=None
                 ):
        """
        :param dropout_ph: None if no dropout should be used. Else a scalar placeholder that determines the prob of dropping a node.
        Remember to set placeholder to Zero during test / eval
        """

        Serializable.quick_init(self, locals())

        with tf.variable_scope(name):
            if input_layer is None:
                l_in = L.InputLayer(shape=(None,) + input_shape, input_var=input_var, name="input")
            else:
                l_in = input_layer
            self._layers = [l_in]
            l_hid = l_in
            if batch_normalization:
                l_hid = L.batch_norm(l_hid)
            for idx, hidden_size in enumerate(hidden_sizes):
                l_hid = L.DenseLayer(
                    l_hid,
                    num_units=hidden_size,
                    nonlinearity=hidden_nonlinearity,
                    name="hidden_%d" % idx,
                    W=hidden_W_init,
                    b=hidden_b_init,
                    weight_normalization=weight_normalization
                )
                if dropout_ph is not None:
                    l_hid = L.DropoutLayer(l_hid, dropout_ph, rescale=False)
                if batch_normalization:
                    l_hid = L.batch_norm(l_hid)
                self._layers.append(l_hid)
            l_out = L.DenseLayer(
                l_hid,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                name="output",
                W=output_W_init,
                b=output_b_init,
                weight_normalization=weight_normalization
            )
            if batch_normalization:
                l_out = L.batch_norm(l_out)
            self._layers.append(l_out)
            self._l_in = l_in
            self._l_out = l_out
            # self._input_var = l_in.input_var
            self._output = L.get_output(l_out)

            LayersPowered.__init__(self, l_out)

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def input_var(self):
        return self._l_in.input_var

    @property
    def layers(self):
        return self._layers

    @property
    def output(self):
        return self._output


class GRUNetwork(object):
    def __init__(self, name, input_shape, output_dim, hidden_dim, hidden_nonlinearity=tf.nn.relu,
                 gru_layer_cls=L.GRULayer,
                 output_nonlinearity=None, input_var=None, input_layer=None, layer_args=None):
        with tf.variable_scope(name):
            if input_layer is None:
                l_in = L.InputLayer(shape=(None, None) + input_shape, input_var=input_var, name="input")
            else:
                l_in = input_layer
            l_step_input = L.InputLayer(shape=(None,) + input_shape, name="step_input")
            l_step_prev_state = L.InputLayer(shape=(None, hidden_dim), name="step_prev_state")
            if layer_args is None:
                layer_args = dict()
            l_gru = gru_layer_cls(l_in, num_units=hidden_dim, hidden_nonlinearity=hidden_nonlinearity,
                                  hidden_init_trainable=False, name="gru", **layer_args)
            l_gru_flat = L.ReshapeLayer(
                l_gru, shape=(-1, hidden_dim),
                name="gru_flat"
            )
            l_output_flat = L.DenseLayer(
                l_gru_flat,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                name="output_flat"
            )
            l_output = L.OpLayer(
                l_output_flat,
                op=lambda flat_output, l_input:
                tf.reshape(flat_output, tf.stack((tf.shape(l_input)[0], tf.shape(l_input)[1], -1))),
                shape_op=lambda flat_output_shape, l_input_shape:
                (l_input_shape[0], l_input_shape[1], flat_output_shape[-1]),
                extras=[l_in],
                name="output"
            )
            l_step_state = l_gru.get_step_layer(l_step_input, l_step_prev_state, name="step_state")
            l_step_hidden = l_step_state
            l_step_output = L.DenseLayer(
                l_step_hidden,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                W=l_output_flat.W,
                b=l_output_flat.b,
                name="step_output"
            )

            self._l_in = l_in
            self._hid_init_param = l_gru.h0
            self._l_gru = l_gru
            self._l_out = l_output
            self._l_step_input = l_step_input
            self._l_step_prev_state = l_step_prev_state
            self._l_step_hidden = l_step_hidden
            self._l_step_state = l_step_state
            self._l_step_output = l_step_output
            self._hidden_dim = hidden_dim

    @property
    def state_dim(self):
        return self._hidden_dim

    @property
    def hidden_dim(self):
        return self._hidden_dim

    @property
    def input_layer(self):
        return self._l_in

    @property
    def input_var(self):
        return self._l_in.input_var

    @property
    def output_layer(self):
        return self._l_out

    @property
    def recurrent_layer(self):
        return self._l_gru

    @property
    def step_input_layer(self):
        return self._l_step_input

    @property
    def step_prev_state_layer(self):
        return self._l_step_prev_state

    @property
    def step_hidden_layer(self):
        return self._l_step_hidden

    @property
    def step_state_layer(self):
        return self._l_step_state

    @property
    def step_output_layer(self):
        return self._l_step_output

    @property
    def hid_init_param(self):
        return self._hid_init_param

    @property
    def state_init_param(self):
        return self._hid_init_param


class LSTMNetwork(object):
    def __init__(self, name, input_shape, output_dim, hidden_dim, hidden_nonlinearity=tf.nn.relu,
                 lstm_layer_cls=L.LSTMLayer,
                 output_nonlinearity=None, input_var=None, input_layer=None, forget_bias=1.0, use_peepholes=False,
                 layer_args=None):
        with tf.variable_scope(name):
            if input_layer is None:
                l_in = L.InputLayer(shape=(None, None) + input_shape, input_var=input_var, name="input")
            else:
                l_in = input_layer
            l_step_input = L.InputLayer(shape=(None,) + input_shape, name="step_input")
            # contains previous hidden and cell state
            l_step_prev_state = L.InputLayer(shape=(None, hidden_dim * 2), name="step_prev_state")
            if layer_args is None:
                layer_args = dict()
            l_lstm = lstm_layer_cls(l_in, num_units=hidden_dim, hidden_nonlinearity=hidden_nonlinearity,
                                    hidden_init_trainable=False, name="lstm", forget_bias=forget_bias,
                                    cell_init_trainable=False, use_peepholes=use_peepholes, **layer_args)
            l_lstm_flat = L.ReshapeLayer(
                l_lstm, shape=(-1, hidden_dim),
                name="lstm_flat"
            )
            l_output_flat = L.DenseLayer(
                l_lstm_flat,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                name="output_flat"
            )
            l_output = L.OpLayer(
                l_output_flat,
                op=lambda flat_output, l_input:
                tf.reshape(flat_output, tf.stack((tf.shape(l_input)[0], tf.shape(l_input)[1], -1))),
                shape_op=lambda flat_output_shape, l_input_shape:
                (l_input_shape[0], l_input_shape[1], flat_output_shape[-1]),
                extras=[l_in],
                name="output"
            )
            l_step_state = l_lstm.get_step_layer(l_step_input, l_step_prev_state, name="step_state")
            l_step_hidden = L.SliceLayer(l_step_state, indices=slice(hidden_dim), name="step_hidden")
            l_step_cell = L.SliceLayer(l_step_state, indices=slice(hidden_dim, None), name="step_cell")
            l_step_output = L.DenseLayer(
                l_step_hidden,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                W=l_output_flat.W,
                b=l_output_flat.b,
                name="step_output"
            )

            self._l_in = l_in
            self._hid_init_param = l_lstm.h0
            self._cell_init_param = l_lstm.c0
            self._l_lstm = l_lstm
            self._l_out = l_output
            self._l_step_input = l_step_input
            self._l_step_prev_state = l_step_prev_state
            self._l_step_hidden = l_step_hidden
            self._l_step_cell = l_step_cell
            self._l_step_state = l_step_state
            self._l_step_output = l_step_output
            self._hidden_dim = hidden_dim

    @property
    def state_dim(self):
        return self._hidden_dim * 2

    @property
    def input_layer(self):
        return self._l_in

    @property
    def input_var(self):
        return self._l_in.input_var

    @property
    def output_layer(self):
        return self._l_out

    @property
    def recurrent_layer(self):
        return self._l_lstm

    @property
    def step_input_layer(self):
        return self._l_step_input

    @property
    def step_prev_state_layer(self):
        return self._l_step_prev_state

    @property
    def step_hidden_layer(self):
        return self._l_step_hidden

    @property
    def step_state_layer(self):
        return self._l_step_state

    @property
    def step_cell_layer(self):
        return self._l_step_cell

    @property
    def step_output_layer(self):
        return self._l_step_output

    @property
    def hid_init_param(self):
        return self._hid_init_param

    @property
    def cell_init_param(self):
        return self._cell_init_param

    @property
    def state_init_param(self):
        return tf.concat(axis=0, values=[self._hid_init_param, self._cell_init_param])
