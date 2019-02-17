import six
import edward as ed
from edward.util import copy, transform
import tensorflow as tf


class MAP_inference(ed.MAP):

  def __init__(self, scope, **kwargs):
    self.scope = scope
    super(MAP_inference, self).__init__(**kwargs)

  def build_loss_and_gradients(self, var_list):
    """Build loss function. Its automatic differentiation
    is the gradient of

    $- \log p(x,z).$
    """
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
    scope = tf.get_default_graph().unique_name("inference")
    dict_swap = {z: qz.value()
                 for z, qz in six.iteritems(self.latent_vars)}
    for x, qx in six.iteritems(self.data):
      if isinstance(x, ed.RandomVariable):
        if isinstance(qx, ed.RandomVariable):
          dict_swap[x] = qx.value()
        else:
          dict_swap[x] = qx

    p_log_prob = 0.0
    for z in six.iterkeys(self.latent_vars):
      z_copy = copy(z, dict_swap, scope=scope)
      p_log_prob += tf.reduce_sum(
          self.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))

    for x in six.iterkeys(self.data):
      if isinstance(x, ed.RandomVariable):
        if dict_swap:
          x_copy = copy(x, dict_swap, scope=scope)
        else:
          x_copy = x
        p_log_prob += tf.reduce_sum(
            self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))

    reg_penalty = tf.reduce_sum(tf.losses.get_regularization_losses(scope=self.scope))
    loss = -p_log_prob + reg_penalty

    grads = tf.gradients(loss, var_list)
    grads_and_vars = list(zip(grads, var_list))
    return loss, grads_and_vars
