import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.util import nest
from tensorflow.python.framework import dtypes


def _check_supported_dtypes(dtype):
  if dtype is None:
    return
  dtype = dtypes.as_dtype(dtype)
  if not (dtype.is_floating or dtype.is_complex):
    raise ValueError("RNN cell only supports floating point inputs, "
                     "but saw dtype: %s" % dtype)


LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple


class CustomLSTM(tf.nn.rnn_cell.LSTMCell):
    def __init__(self, num_units, stop_gradients=False, **kwargs):
        super(CustomLSTM, self).__init__(num_units, *kwargs)
        self.stop_gradients = stop_gradients

    def call(self, inputs, state):
        """Run one step of LSTM.

        Args:
          inputs: input Tensor, must be 2-D, `[batch, input_size]`.
          state: if `state_is_tuple` is False, this must be a state Tensor, `2-D,
            [batch, state_size]`.  If `state_is_tuple` is True, this must be a tuple
            of state Tensors, both `2-D`, with column sizes `c_state` and `m_state`.

        Returns:
          A tuple containing:

          - A `2-D, [batch, output_dim]`, Tensor representing the output of the
            LSTM after reading `inputs` when previous state was `state`.
            Here output_dim is:
               num_proj if num_proj was set,
               num_units otherwise.
          - Tensor(s) representing the new state of LSTM after reading `inputs` when
            the previous state was `state`.  Same type and shape(s) as `state`.

        Raises:
          ValueError: If input size cannot be inferred from inputs via
            static shape inference.
        """
        for t in nest.flatten([inputs, state]):
            _check_supported_dtypes(t.dtype)

        num_proj = self._num_units if self._num_proj is None else self._num_proj
        sigmoid = math_ops.sigmoid

        if self._state_is_tuple:
            (c_prev, m_prev) = state
        else:
            c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
            m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

        if self.stop_gradients:
            m_prev = tf.stop_gradient(m_prev)
            inputs = tf.stop_gradient(inputs)

        input_size = inputs.get_shape().with_rank(2).dims[1].value
        if input_size is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        lstm_matrix = math_ops.matmul(
            array_ops.concat([inputs, m_prev], 1), self._kernel)
        lstm_matrix = nn_ops.bias_add(lstm_matrix, self._bias)

        i, j, f, o = array_ops.split(
            value=lstm_matrix, num_or_size_splits=4, axis=1)
        # Diagonal connections
        if self._use_peepholes:
            c = (
                    sigmoid(f + self._forget_bias + self._w_f_diag * c_prev) * c_prev +
                    sigmoid(i + self._w_i_diag * c_prev) * self._activation(j))
        else:
            c = (
                    sigmoid(f + self._forget_bias) * c_prev +
                    sigmoid(i) * self._activation(j))

        if self._cell_clip is not None:
            # pylint: disable=invalid-unary-operand-type
            c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
            # pylint: enable=invalid-unary-operand-type
        if self._use_peepholes:
            m = sigmoid(o + self._w_o_diag * c) * self._activation(c)
        else:
            m = sigmoid(o) * self._activation(c)

        if self._num_proj is not None:
            m = math_ops.matmul(m, self._proj_kernel)

            if self._proj_clip is not None:
                # pylint: disable=invalid-unary-operand-type
                m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
                # pylint: enable=invalid-unary-operand-type

        new_state = (
            LSTMStateTuple(c, m)
            if self._state_is_tuple else array_ops.concat([c, m], 1))
        return m, new_state

