from unittest import TestCase

from math import e
import numpy as np
import deepchem
from deepchem.models.tensorgraph.models.rave import RaveModel, _kl_divergence, _linear_scale


def three_state_potential(x, y):
  e1 = -1 * (x + 1)**2 - 2 * (y - 1)**2
  e2 = -2 * (x + 0.8)**2 - 2 * (y + 1)**2
  e3 = -2 * (x - 1)**2 - 2 * y**2
  vs = [-12 * e**x for x in [e1, e2, e3]]
  return min(vs)


class TestRaveModel(TestCase):

  def create_dataset(self, n_features=3, n_samples=100):
    X = np.random.random(size=(n_samples, n_features))
    w = np.ones(shape=(n_samples, n_features))
    return deepchem.data.NumpyDataset(X, X, w)

  def test_rave_model(self):
    n_features = 3
    model = RaveModel(
        n_features=n_features,
        use_queue=False,
        kl_annealing_start_step=0,
        kl_annealing_stop_step=0)
    ds = self.create_dataset(n_features)

    model.fit(ds, nb_epoch=1000)

    means = model.predict(ds, outputs=model.mean)
    reconstructions = model.predict(ds, outputs=model.reconstruction)
    model.save()

    model = RaveModel.load_from_dir(model.model_dir)

    m2 = model.predict(ds, outputs=model.mean)
    r2 = model.predict(ds, outputs=model.reconstruction)
    self.assertTrue(np.all(means == m2))
    self.assertTrue(np.all(reconstructions == r2))

  def test_kl_divergence(self):
    n_samples = 100
    d1 = np.random.random(size=(n_samples,))
    d2 = np.random.random(size=(n_samples,))

    retval = _kl_divergence(d1, d2)
    self.assertTrue(retval > 0)

    retval = _kl_divergence(d1, d1)
    self.assertTrue(np.isclose(0, retval))

  def test_linear_scale(self):
    d1 = np.random.random(size=(200,))
    d2 = _linear_scale(d1)
    my_min = np.min(d2)
    my_max = np.max(d2)
    self.assertEqual(0, my_min)
    self.assertEqual(1, my_max)
