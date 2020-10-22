"""
Test reload for trained models.
"""
import os
import tempfile

import scipy
import numpy as np
import tensorflow as tf
from flaky import flaky
from sklearn.ensemble import RandomForestClassifier

import deepchem as dc
from deepchem.feat import create_char_to_idx
from deepchem.molnet.load_function.chembl25_datasets import chembl25_tasks


def test_sklearn_classifier_reload():
  """Test that trained model can be reloaded correctly."""
  n_samples = 10
  n_features = 3
  n_tasks = 1

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.random.randint(2, size=(n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))

  dataset = dc.data.NumpyDataset(X, y, w, ids)
  classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score)

  sklearn_model = RandomForestClassifier()
  model_dir = tempfile.mkdtemp()
  model = dc.models.SklearnModel(sklearn_model, model_dir)

  # Fit trained model
  model.fit(dataset)
  model.save()

  # Load trained model
  reloaded_model = dc.models.SklearnModel(None, model_dir)
  reloaded_model.reload()

  # Check predictions match on random sample
  Xpred = np.random.rand(n_samples, n_features)
  predset = dc.data.NumpyDataset(Xpred)
  origpred = model.predict(predset)
  reloadpred = reloaded_model.predict(predset)
  assert np.all(origpred == reloadpred)

  # Eval model on train
  scores = reloaded_model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .9


def test_multitaskregressor_reload():
  """Test that MultitaskRegressor can be reloaded correctly."""
  n_samples = 10
  n_features = 3
  n_tasks = 1

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.random.rand(n_samples, n_tasks)
  w = np.ones((n_samples, n_tasks))

  dataset = dc.data.NumpyDataset(X, y, w, ids)
  regression_metric = dc.metrics.Metric(dc.metrics.mean_squared_error)

  model_dir = tempfile.mkdtemp()
  model = dc.models.MultitaskRegressor(
      n_tasks,
      n_features,
      dropouts=[0.],
      weight_init_stddevs=[np.sqrt(6) / np.sqrt(1000)],
      batch_size=n_samples,
      learning_rate=0.003,
      model_dir=model_dir)

  # Fit trained model
  model.fit(dataset, nb_epoch=100)

  # Eval model on train
  scores = model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] < .1

  # Reload trained model
  reloaded_model = dc.models.MultitaskRegressor(
      n_tasks,
      n_features,
      dropouts=[0.],
      weight_init_stddevs=[np.sqrt(6) / np.sqrt(1000)],
      batch_size=n_samples,
      learning_rate=0.003,
      model_dir=model_dir)
  reloaded_model.restore()

  # Check predictions match on random sample
  Xpred = np.random.rand(n_samples, n_features)
  predset = dc.data.NumpyDataset(Xpred)
  origpred = model.predict(predset)
  reloadpred = reloaded_model.predict(predset)
  assert np.all(origpred == reloadpred)

  # Eval model on train
  scores = reloaded_model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] < 0.1


def test_multitaskclassification_reload():
  """Test that MultitaskClassifier can be reloaded correctly."""
  n_samples = 10
  n_features = 3
  n_tasks = 1

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.zeros((n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))
  dataset = dc.data.NumpyDataset(X, y, w, ids)

  classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score)
  model_dir = tempfile.mkdtemp()
  model = dc.models.MultitaskClassifier(
      n_tasks,
      n_features,
      dropouts=[0.],
      weight_init_stddevs=[.1],
      batch_size=n_samples,
      optimizer=dc.models.optimizers.Adam(
          learning_rate=0.0003, beta1=0.9, beta2=0.999),
      model_dir=model_dir)

  # Fit trained model
  model.fit(dataset, nb_epoch=100)

  # Reload trained model
  reloaded_model = dc.models.MultitaskClassifier(
      n_tasks,
      n_features,
      dropouts=[0.],
      weight_init_stddevs=[.1],
      batch_size=n_samples,
      optimizer=dc.models.optimizers.Adam(
          learning_rate=0.0003, beta1=0.9, beta2=0.999),
      model_dir=model_dir)
  reloaded_model.restore()

  # Check predictions match on random sample
  Xpred = np.random.rand(n_samples, n_features)
  predset = dc.data.NumpyDataset(Xpred)
  origpred = model.predict(predset)
  reloadpred = reloaded_model.predict(predset)
  assert np.all(origpred == reloadpred)

  # Eval model on train
  scores = reloaded_model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .9


def test_residual_classification_reload():
  """Test that a residual network can reload correctly."""
  n_samples = 10
  n_features = 5
  n_tasks = 1

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.random.randint(2, size=(n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))
  dataset = dc.data.NumpyDataset(X, y, w, ids)

  classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score)
  model_dir = tempfile.mkdtemp()
  model = dc.models.MultitaskClassifier(
      n_tasks,
      n_features,
      layer_sizes=[20] * 10,
      dropouts=0.0,
      batch_size=n_samples,
      residual=True,
      model_dir=model_dir)

  # Fit trained model
  model.fit(dataset, nb_epoch=500)

  # Eval model on train
  scores = model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .9

  # Reload trained model
  reloaded_model = dc.models.MultitaskClassifier(
      n_tasks,
      n_features,
      layer_sizes=[20] * 10,
      dropouts=0.0,
      batch_size=n_samples,
      residual=True,
      model_dir=model_dir)
  reloaded_model.restore()

  # Check predictions match on random sample
  Xpred = np.random.rand(n_samples, n_features)
  predset = dc.data.NumpyDataset(Xpred)
  origpred = model.predict(predset)
  reloadpred = reloaded_model.predict(predset)
  assert np.all(origpred == reloadpred)

  # Eval model on train
  scores = reloaded_model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .9


def test_robust_multitask_classification_reload():
  """Test robust multitask overfits tiny data."""
  n_tasks = 10
  n_samples = 10
  n_features = 3

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.zeros((n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))
  dataset = dc.data.NumpyDataset(X, y, w, ids)

  classification_metric = dc.metrics.Metric(
      dc.metrics.accuracy_score, task_averager=np.mean)
  model_dir = tempfile.mkdtemp()
  model = dc.models.RobustMultitaskClassifier(
      n_tasks,
      n_features,
      layer_sizes=[50],
      bypass_layer_sizes=[10],
      dropouts=[0.],
      learning_rate=0.003,
      weight_init_stddevs=[.1],
      batch_size=n_samples,
      model_dir=model_dir)

  # Fit trained model
  model.fit(dataset, nb_epoch=25)

  # Eval model on train
  scores = model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .9

  # Reloaded Trained Model
  reloaded_model = dc.models.RobustMultitaskClassifier(
      n_tasks,
      n_features,
      layer_sizes=[50],
      bypass_layer_sizes=[10],
      dropouts=[0.],
      learning_rate=0.003,
      weight_init_stddevs=[.1],
      batch_size=n_samples,
      model_dir=model_dir)
  reloaded_model.restore()

  # Check predictions match on random sample
  Xpred = np.random.rand(n_samples, n_features)
  predset = dc.data.NumpyDataset(Xpred)
  origpred = model.predict(predset)
  reloadpred = reloaded_model.predict(predset)
  assert np.all(origpred == reloadpred)

  # Eval model on train
  scores = reloaded_model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .9


def test_robust_multitask_regressor_reload():
  """Test that RobustMultitaskRegressor can be reloaded correctly."""
  n_tasks = 10
  n_samples = 10
  n_features = 3

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.random.rand(n_samples, n_tasks)
  w = np.ones((n_samples, n_tasks))

  dataset = dc.data.NumpyDataset(X, y, w, ids)
  regression_metric = dc.metrics.Metric(dc.metrics.mean_squared_error)

  model_dir = tempfile.mkdtemp()
  model = dc.models.RobustMultitaskRegressor(
      n_tasks,
      n_features,
      layer_sizes=[50],
      bypass_layer_sizes=[10],
      dropouts=[0.],
      learning_rate=0.003,
      weight_init_stddevs=[.1],
      batch_size=n_samples,
      model_dir=model_dir)

  # Fit trained model
  model.fit(dataset, nb_epoch=100)

  # Eval model on train
  scores = model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] < .1

  # Reload trained model
  reloaded_model = dc.models.RobustMultitaskRegressor(
      n_tasks,
      n_features,
      layer_sizes=[50],
      bypass_layer_sizes=[10],
      dropouts=[0.],
      learning_rate=0.003,
      weight_init_stddevs=[.1],
      batch_size=n_samples,
      model_dir=model_dir)
  reloaded_model.restore()

  # Check predictions match on random sample
  Xpred = np.random.rand(n_samples, n_features)
  predset = dc.data.NumpyDataset(Xpred)
  origpred = model.predict(predset)
  reloadpred = reloaded_model.predict(predset)
  assert np.all(origpred == reloadpred)

  # Eval model on train
  scores = reloaded_model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] < 0.1


def test_IRV_multitask_classification_reload():
  """Test IRV classifier can be reloaded."""
  n_tasks = 5
  n_samples = 10
  n_features = 128

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.randint(2, size=(n_samples, n_features))
  y = np.ones((n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))
  dataset = dc.data.NumpyDataset(X, y, w, ids)
  IRV_transformer = dc.trans.IRVTransformer(5, n_tasks, dataset)
  dataset_trans = IRV_transformer.transform(dataset)

  classification_metric = dc.metrics.Metric(
      dc.metrics.accuracy_score, task_averager=np.mean)
  model_dir = tempfile.mkdtemp()
  model = dc.models.MultitaskIRVClassifier(
      n_tasks,
      K=5,
      learning_rate=0.01,
      batch_size=n_samples,
      model_dir=model_dir)

  # Fit trained model
  model.fit(dataset_trans)

  # Eval model on train
  scores = model.evaluate(dataset_trans, [classification_metric])
  assert scores[classification_metric.name] > .9

  # Reload Trained Model
  reloaded_model = dc.models.MultitaskIRVClassifier(
      n_tasks,
      K=5,
      learning_rate=0.01,
      batch_size=n_samples,
      model_dir=model_dir)
  reloaded_model.restore()

  # Check predictions match on random sample
  Xpred = np.random.rand(n_samples, n_features)
  predset = dc.data.NumpyDataset(Xpred)
  origpred = model.predict(predset)
  reloadpred = reloaded_model.predict(predset)
  assert np.all(origpred == reloadpred)

  # Eval model on train
  scores = reloaded_model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .9


@flaky
def test_progressive_classification_reload():
  """Test progressive multitask can reload."""
  np.random.seed(123)
  n_tasks = 5
  n_samples = 10
  n_features = 6

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.random.randint(2, size=(n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))

  dataset = dc.data.NumpyDataset(X, y, w, ids)

  classification_metric = dc.metrics.Metric(
      dc.metrics.accuracy_score, task_averager=np.mean)
  model_dir = tempfile.mkdtemp()
  model = dc.models.ProgressiveMultitaskClassifier(
      n_tasks,
      n_features,
      layer_sizes=[50],
      bypass_layer_sizes=[10],
      dropouts=[0.],
      learning_rate=0.001,
      weight_init_stddevs=[.1],
      alpha_init_stddevs=[.02],
      batch_size=n_samples,
      model_dir=model_dir)

  # Fit trained model
  model.fit(dataset, nb_epoch=400)

  # Eval model on train
  scores = model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .9

  # Reload Trained Model
  reloaded_model = dc.models.ProgressiveMultitaskClassifier(
      n_tasks,
      n_features,
      layer_sizes=[50],
      bypass_layer_sizes=[10],
      dropouts=[0.],
      learning_rate=0.001,
      weight_init_stddevs=[.1],
      alpha_init_stddevs=[.02],
      batch_size=n_samples,
      model_dir=model_dir)
  reloaded_model.restore()

  # Check predictions match on random sample
  Xpred = np.random.rand(n_samples, n_features)
  predset = dc.data.NumpyDataset(Xpred)
  origpred = model.predict(predset)
  reloadpred = reloaded_model.predict(predset)
  assert np.all(origpred == reloadpred)

  # Eval model on train
  scores = reloaded_model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .9


def test_progressivemultitaskregressor_reload():
  """Test that ProgressiveMultitaskRegressor can be reloaded correctly."""
  n_samples = 10
  n_features = 3
  n_tasks = 1

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.random.rand(n_samples, n_tasks)
  w = np.ones((n_samples, n_tasks))

  dataset = dc.data.NumpyDataset(X, y, w, ids)
  regression_metric = dc.metrics.Metric(dc.metrics.mean_squared_error)

  model_dir = tempfile.mkdtemp()
  model = dc.models.ProgressiveMultitaskRegressor(
      n_tasks,
      n_features,
      layer_sizes=[50],
      bypass_layer_sizes=[10],
      dropouts=[0.],
      learning_rate=0.001,
      weight_init_stddevs=[.1],
      alpha_init_stddevs=[.02],
      batch_size=n_samples,
      model_dir=model_dir)

  # Fit trained model
  model.fit(dataset, nb_epoch=100)

  # Eval model on train
  scores = model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] < .1

  # Reload trained model
  reloaded_model = dc.models.ProgressiveMultitaskRegressor(
      n_tasks,
      n_features,
      layer_sizes=[50],
      bypass_layer_sizes=[10],
      dropouts=[0.],
      learning_rate=0.001,
      weight_init_stddevs=[.1],
      alpha_init_stddevs=[.02],
      batch_size=n_samples,
      model_dir=model_dir)
  reloaded_model.restore()

  # Check predictions match on random sample
  Xpred = np.random.rand(n_samples, n_features)
  predset = dc.data.NumpyDataset(Xpred)
  origpred = model.predict(predset)
  reloadpred = reloaded_model.predict(predset)
  assert np.all(origpred == reloadpred)

  # Eval model on train
  scores = reloaded_model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] < 0.1


def test_DAG_regression_reload():
  """Test DAG regressor reloads."""
  np.random.seed(123)
  tf.random.set_seed(123)
  n_tasks = 1

  # Load mini log-solubility dataset.
  featurizer = dc.feat.ConvMolFeaturizer()
  mols = ["CC", "CCO", "CC", "CCC", "CCCCO", "CO", "CC", "CCCCC", "CCC", "CCCO"]
  n_samples = len(mols)
  X = featurizer(mols)
  y = np.random.rand(n_samples, n_tasks)
  dataset = dc.data.NumpyDataset(X, y)

  regression_metric = dc.metrics.Metric(
      dc.metrics.pearson_r2_score, task_averager=np.mean)

  n_feat = 75
  batch_size = 10
  transformer = dc.trans.DAGTransformer(max_atoms=50)
  dataset = transformer.transform(dataset)

  model_dir = tempfile.mkdtemp()
  model = dc.models.DAGModel(
      n_tasks,
      max_atoms=50,
      n_atom_feat=n_feat,
      batch_size=batch_size,
      learning_rate=0.001,
      use_queue=False,
      mode="regression",
      model_dir=model_dir)

  # Fit trained model
  model.fit(dataset, nb_epoch=100)

  # Eval model on train
  scores = model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] > .1

  reloaded_model = dc.models.DAGModel(
      n_tasks,
      max_atoms=50,
      n_atom_feat=n_feat,
      batch_size=batch_size,
      learning_rate=0.001,
      use_queue=False,
      mode="regression",
      model_dir=model_dir)

  reloaded_model.restore()

  # Check predictions match on random sample
  predmols = ["CCCC", "CCCCCO", "CCCCC"]
  Xpred = featurizer(predmols)
  predset = dc.data.NumpyDataset(Xpred)
  predset = transformer.transform(predset)
  origpred = model.predict(predset)
  reloadpred = reloaded_model.predict(predset)

  assert np.all(origpred == reloadpred)

  # Eval model on train
  scores = reloaded_model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] > .1


def test_weave_classification_reload():
  """Test weave model can be reloaded."""
  np.random.seed(123)
  tf.random.set_seed(123)
  n_tasks = 1

  # Load mini log-solubility dataset.
  featurizer = dc.feat.WeaveFeaturizer()
  mols = ["CC", "CCCCC", "CCCCC", "CCC", "COOO", "COO", "OO"]
  X = featurizer(mols)
  y = [1, 1, 1, 1, 0, 0, 0]
  dataset = dc.data.NumpyDataset(X, y)

  classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score)

  batch_size = 5

  model_dir = tempfile.mkdtemp()
  model = dc.models.WeaveModel(
      n_tasks,
      batch_size=batch_size,
      learning_rate=0.01,
      mode="classification",
      dropouts=0.0,
      model_dir=model_dir)

  # Fit trained model
  model.fit(dataset, nb_epoch=100)

  # Eval model on train
  scores = model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .6

  # Check predictions match on random sample
  predmols = ["CCCC", "CCCCCO", "CCCCC"]
  Xpred = featurizer(predmols)

  predset = dc.data.NumpyDataset(Xpred)
  origpred = model.predict(predset)

  reloaded_model = dc.models.WeaveModel(
      n_tasks,
      batch_size=batch_size,
      learning_rate=0.003,
      mode="classification",
      dropouts=0.0,
      model_dir=model_dir)
  reloaded_model.restore()

  # Check predictions match on random sample
  predmols = ["CCCC", "CCCCCO", "CCCCC"]
  Xpred = featurizer(predmols)
  predset = dc.data.NumpyDataset(Xpred)
  origpred = model.predict(predset)
  reloadpred = reloaded_model.predict(predset)
  assert np.all(origpred == reloadpred)

  # Eval model on train
  scores = reloaded_model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .6


def test_MPNN_regression_reload():
  """Test MPNN can reload datasets."""
  np.random.seed(123)
  tf.random.set_seed(123)
  n_tasks = 1

  # Load mini log-solubility dataset.
  featurizer = dc.feat.WeaveFeaturizer()
  mols = ["C", "CO", "CC"]
  n_samples = len(mols)
  X = featurizer(mols)
  y = np.random.rand(n_samples, n_tasks)
  dataset = dc.data.NumpyDataset(X, y)

  regression_metric = dc.metrics.Metric(
      dc.metrics.pearson_r2_score, task_averager=np.mean)

  n_atom_feat = 75
  n_pair_feat = 14
  batch_size = 10
  model_dir = tempfile.mkdtemp()
  model = dc.models.MPNNModel(
      n_tasks,
      n_atom_feat=n_atom_feat,
      n_pair_feat=n_pair_feat,
      T=2,
      M=3,
      batch_size=batch_size,
      learning_rate=0.001,
      use_queue=False,
      mode="regression",
      model_dir=model_dir)

  # Fit trained model
  model.fit(dataset, nb_epoch=50)

  # Eval model on train
  scores = model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] > .8

  # Reload trained model
  reloaded_model = dc.models.MPNNModel(
      n_tasks,
      n_atom_feat=n_atom_feat,
      n_pair_feat=n_pair_feat,
      T=2,
      M=3,
      batch_size=batch_size,
      learning_rate=0.001,
      use_queue=False,
      mode="regression",
      model_dir=model_dir)
  reloaded_model.restore()

  # Eval model on train
  scores = reloaded_model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] > .8

  # Check predictions match on random sample
  predmols = ["CCCC", "CCCCCO", "CCCCC"]
  Xpred = featurizer(predmols)
  predset = dc.data.NumpyDataset(Xpred)
  origpred = model.predict(predset)
  reloadpred = reloaded_model.predict(predset)
  assert np.all(origpred == reloadpred)


def test_textCNN_classification_reload():
  """Test textCNN model reloadinng."""
  np.random.seed(123)
  tf.random.set_seed(123)
  n_tasks = 1

  featurizer = dc.feat.RawFeaturizer()
  mols = ["C", "CO", "CC"]
  n_samples = len(mols)
  X = featurizer(mols)
  y = np.random.randint(2, size=(n_samples, n_tasks))
  dataset = dc.data.NumpyDataset(X, y, ids=mols)

  classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score)

  char_dict, length = dc.models.TextCNNModel.build_char_dict(dataset)
  batch_size = 3

  model_dir = tempfile.mkdtemp()
  model = dc.models.TextCNNModel(
      n_tasks,
      char_dict,
      seq_length=length,
      batch_size=batch_size,
      learning_rate=0.001,
      use_queue=False,
      mode="classification",
      model_dir=model_dir)

  # Fit trained model
  model.fit(dataset, nb_epoch=200)

  # Eval model on train
  scores = model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .8

  # Reload trained model
  reloaded_model = dc.models.TextCNNModel(
      n_tasks,
      char_dict,
      seq_length=length,
      batch_size=batch_size,
      learning_rate=0.001,
      use_queue=False,
      mode="classification",
      model_dir=model_dir)
  reloaded_model.restore()

  # Eval model on train
  scores = reloaded_model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .8

  assert len(reloaded_model.model.get_weights()) == len(
      model.model.get_weights())
  for (reloaded, orig) in zip(reloaded_model.model.get_weights(),
                              model.model.get_weights()):
    assert np.all(reloaded == orig)

  # Check predictions match on random sample
  predmols = ["CCCC", "CCCCCO", "CCCCC"]
  Xpred = featurizer(predmols)
  predset = dc.data.NumpyDataset(Xpred, ids=predmols)
  origpred = model.predict(predset)
  reloadpred = reloaded_model.predict(predset)
  assert np.all(origpred == reloadpred)

  assert len(model.model.layers) == len(reloaded_model.model.layers)


def test_1d_cnn_regression_reload():
  """Test that a 1D CNN can reload."""
  n_samples = 10
  n_features = 3
  n_tasks = 1

  np.random.seed(123)
  X = np.random.rand(n_samples, 10, n_features)
  y = np.random.randint(2, size=(n_samples, n_tasks)).astype(np.float32)
  dataset = dc.data.NumpyDataset(X, y)

  regression_metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
  model_dir = tempfile.mkdtemp()

  model = dc.models.CNN(
      n_tasks,
      n_features,
      dims=1,
      dropouts=0,
      kernel_size=3,
      mode='regression',
      learning_rate=0.003,
      model_dir=model_dir)

  # Fit trained model
  model.fit(dataset, nb_epoch=200)

  # Eval model on train
  scores = model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] < 0.1

  # Reload trained model
  reloaded_model = dc.models.CNN(
      n_tasks,
      n_features,
      dims=1,
      dropouts=0,
      kernel_size=3,
      mode='regression',
      learning_rate=0.003,
      model_dir=model_dir)
  reloaded_model.restore()

  # Check predictions match on random sample
  Xpred = np.random.rand(n_samples, 10, n_features)
  predset = dc.data.NumpyDataset(Xpred)
  origpred = model.predict(predset)
  reloadpred = reloaded_model.predict(predset)
  assert np.all(origpred == reloadpred)

  # Eval model on train
  scores = reloaded_model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] < 0.1


def test_graphconvmodel_reload():
  featurizer = dc.feat.ConvMolFeaturizer()
  tasks = ["outcome"]
  mols = ["C", "CO", "CC"]
  X = featurizer(mols)
  y = np.array([0, 1, 0])
  dataset = dc.data.NumpyDataset(X, y)

  classification_metric = dc.metrics.Metric(
      dc.metrics.roc_auc_score, np.mean, mode="classification")

  batch_size = 10
  model_dir = tempfile.mkdtemp()
  model = dc.models.GraphConvModel(
      len(tasks),
      batch_size=batch_size,
      batch_normalize=False,
      mode='classification',
      model_dir=model_dir)

  model.fit(dataset, nb_epoch=10)
  scores = model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] >= 0.6

  # Reload trained Model
  reloaded_model = dc.models.GraphConvModel(
      len(tasks),
      batch_size=batch_size,
      batch_normalize=False,
      mode='classification',
      model_dir=model_dir)
  reloaded_model.restore()

  # Check predictions match on random sample
  predmols = ["CCCC", "CCCCCO", "CCCCC"]
  Xpred = featurizer(predmols)
  predset = dc.data.NumpyDataset(Xpred)
  origpred = model.predict(predset)
  reloadpred = reloaded_model.predict(predset)
  assert np.all(origpred == reloadpred)

  # Eval model on train
  scores = reloaded_model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .6


def test_chemception_reload():
  """Test that chemception models can be saved and reloaded."""
  img_size = 80
  img_spec = "engd"
  res = 0.5
  n_tasks = 1
  featurizer = dc.feat.SmilesToImage(
      img_size=img_size, img_spec=img_spec, res=res)

  data_points = 10
  mols = ["CCCCCCCC"] * data_points
  X = featurizer(mols)

  y = np.random.randint(0, 2, size=(data_points, n_tasks))
  w = np.ones(shape=(data_points, n_tasks))
  dataset = dc.data.NumpyDataset(X, y, w, mols)

  model_dir = tempfile.mkdtemp()
  model = dc.models.ChemCeption(
      n_tasks=n_tasks,
      img_spec="engd",
      model_dir=model_dir,
      mode="classification")
  model.fit(dataset, nb_epoch=3)

  # Reload Trained Model
  reloaded_model = dc.models.ChemCeption(
      n_tasks=n_tasks,
      img_spec="engd",
      model_dir=model_dir,
      mode="classification")
  reloaded_model.restore()

  # Check predictions match on random sample
  predmols = ["CCCC", "CCCCCO", "CCCCC"]
  Xpred = featurizer(predmols)
  predset = dc.data.NumpyDataset(Xpred)
  origpred = model.predict(predset)
  reloadpred = reloaded_model.predict(predset)
  assert np.all(origpred == reloadpred)


# TODO: This test is a little awkward. The Smiles2Vec model awkwardly depends on a dataset_file being available on disk.
# This needs to be cleaned up to match the standard model handling API.
def test_smiles2vec_reload():
  """Test that smiles2vec models can be saved and reloaded."""
  dataset_file = os.path.join(os.path.dirname(__file__), "chembl_25_small.csv")
  max_len = 250
  pad_len = 10
  max_seq_len = 20
  char_to_idx = create_char_to_idx(
      dataset_file, max_len=max_len, smiles_field="smiles")
  feat = dc.feat.SmilesToSeq(
      char_to_idx=char_to_idx, max_len=max_len, pad_len=pad_len)

  n_tasks = 5
  data_points = 10

  loader = dc.data.CSVLoader(
      tasks=chembl25_tasks, smiles_field='smiles', featurizer=feat)
  dataset = loader.create_dataset(
      inputs=[dataset_file], shard_size=10000, data_dir=tempfile.mkdtemp())
  y = np.random.randint(0, 2, size=(data_points, n_tasks))
  w = np.ones(shape=(data_points, n_tasks))
  dataset = dc.data.NumpyDataset(dataset.X[:data_points, :max_seq_len], y, w,
                                 dataset.ids[:data_points])

  model_dir = tempfile.mkdtemp()
  model = dc.models.Smiles2Vec(
      char_to_idx=char_to_idx,
      max_seq_len=max_seq_len,
      use_conv=True,
      n_tasks=n_tasks,
      model_dir=model_dir,
      mode="classification")
  model.fit(dataset, nb_epoch=3)

  # Reload Trained Model
  reloaded_model = dc.models.Smiles2Vec(
      char_to_idx=char_to_idx,
      max_seq_len=max_seq_len,
      use_conv=True,
      n_tasks=n_tasks,
      model_dir=model_dir,
      mode="classification")
  reloaded_model.restore()

  # Check predictions match on original dataset
  origpred = model.predict(dataset)
  reloadpred = reloaded_model.predict(dataset)
  assert np.all(origpred == reloadpred)


# TODO: We need a cleaner usage example for this
def test_DTNN_regression_reload():
  """Test DTNN can reload datasets."""
  np.random.seed(123)
  tf.random.set_seed(123)
  n_tasks = 1

  current_dir = os.path.dirname(os.path.abspath(__file__))
  input_file = os.path.join(current_dir, "example_DTNN.mat")
  dataset = scipy.io.loadmat(input_file)
  X = dataset['X']
  y = dataset['T']
  w = np.ones_like(y)
  dataset = dc.data.NumpyDataset(X, y, w, ids=None)
  n_tasks = y.shape[1]

  model_dir = tempfile.mkdtemp()
  model = dc.models.DTNNModel(
      n_tasks,
      n_embedding=20,
      n_distance=100,
      learning_rate=1.0,
      model_dir=model_dir,
      mode="regression")

  # Fit trained model
  model.fit(dataset, nb_epoch=250)

  # Eval model on train
  pred = model.predict(dataset)
  mean_rel_error = np.mean(np.abs(1 - pred / y))
  assert mean_rel_error < 0.2

  reloaded_model = dc.models.DTNNModel(
      n_tasks,
      n_embedding=20,
      n_distance=100,
      learning_rate=1.0,
      model_dir=model_dir,
      mode="regression")
  reloaded_model.restore()

  # Check predictions match on random sample
  origpred = model.predict(dataset)
  reloadpred = reloaded_model.predict(dataset)
  assert np.all(origpred == reloadpred)
