import tensorflow as tf
from deepchem.models import TensorGraph
from deepchem.models.tensorgraph.layers import Feature, CombineMeanStd, Weights, Dense, L2Loss, KLDivergenceLoss, Add, \
  TensorWrapper, ReduceSum

import numpy as np


class RaveModel(TensorGraph):

  def __init__(self,
               n_features,
               encoder_layers=[512, 512, 521],
               decoder_layers=[512, 512, 512],
               kl_annealing_start_step=500,
               kl_annealing_stop_step=1000,
               **kwargs):
    self.n_features = n_features
    self.encoder_layers = encoder_layers
    self.decoder_layers = decoder_layers
    self.kl_annealing_start_step = kl_annealing_start_step
    self.kl_annealing_stop_step = kl_annealing_stop_step
    super(RaveModel, self).__init__(**kwargs)

    self.build_graph()

  def build_graph(self):
    print("building")
    features = Feature(shape=(None, self.n_features))
    last_layer = features
    for layer_size in self.encoder_layers:
      last_layer = Dense(
          in_layers=last_layer,
          activation_fn=tf.nn.elu,
          out_channels=layer_size)

    self.mean = Dense(in_layers=last_layer, activation_fn=None, out_channels=1)
    self.std = Dense(in_layers=last_layer, activation_fn=None, out_channels=1)

    readout = CombineMeanStd([self.mean, self.std], training_only=True)
    last_layer = readout
    for layer_size in self.decoder_layers:
      last_layer = Dense(
          in_layers=readout, activation_fn=tf.nn.elu, out_channels=layer_size)

    self.reconstruction = Dense(
        in_layers=last_layer, activation_fn=None, out_channels=self.n_features)
    weights = Weights(shape=(None, self.n_features))
    reproduction_loss = L2Loss(
        in_layers=[features, self.reconstruction, weights])
    reproduction_loss = ReduceSum(in_layers=reproduction_loss, axis=0)
    global_step = TensorWrapper(self._get_tf("GlobalStep"))
    kl_loss = KLDivergenceLoss(
        in_layers=[self.mean, self.std, global_step],
        annealing_start_step=self.kl_annealing_start_step,
        annealing_stop_step=self.kl_annealing_stop_step)
    loss = Add(in_layers=[kl_loss, reproduction_loss], weights=[0.5, 1])

    self.add_output(self.mean)
    self.add_output(self.reconstruction)
    self.set_loss(loss)

  def save(self):
    super(RaveModel, self).save()
    self._save_kwargs({
        "n_features": self.n_features,
        "encoder_layers": self.encoder_layers,
        "decoder_layers": self.decoder_layers,
        "kl_annealing_start_step": self.kl_annealing_start_step,
        "kl_annealing_stop_step": self.kl_annealing_stop_step,
    })


def _linear_scale(d):
  my_min = np.min(d)
  d -= my_min
  my_max = np.max(d)
  d /= my_max
  return d


def _kl_divergence(d1, d2, n_bins=10):
  d1, d2 = _linear_scale(d1), _linear_scale(d2)
  n = len(d1)
  bins = np.linspace(0, 1, n_bins)
  d1 = np.digitize(d1, bins, right=True)
  d2 = np.digitize(d2, bins, right=True)

  total = 0
  for i in range(1, n_bins + 1):
    p_d1i = np.sum(d1 == i) / n
    p_d2i = np.sum(d2 == i) / n + np.finfo(float).eps
    total += p_d1i * np.log(p_d1i / p_d2i + np.finfo(float).eps)
  return total


def find_rc(rave_model, ds):
  pass
