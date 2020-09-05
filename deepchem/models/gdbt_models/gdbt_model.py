"""
Scikit-learn wrapper interface
"""

import os
import logging
import tempfile
from typing import Callable, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from deepchem.data import Dataset
from deepchem.models.sklearn_models import SklearnModel

logger = logging.getLogger(__name__)


class GDBTModel(SklearnModel):
  """Wrapper class that wraps GDBT models as DeepChem models.

  This class supports LightGBM/XGBoost models.
  """

  def __init__(self,
               model_instance: BaseEstimator,
               model_dir: Optional[str] = None,
               early_stopping_rounds: int = 50,
               eval_metric: Optional[Union[str, Callable[..., Tuple]]] = None,
               **kwargs):
    """
    Parameters
    ----------
    model_instance: BaseEstimator
      The model instance of scikit-learn wrapper LightGBM/XGBoost models.
    model_dir: str, optional (default None)
      Path to directory where model will be stored.
    early_stopping_rounds: int, optional (default 50)
      Activates early stopping. Validation metric needs to improve at least once
      in every early_stopping_rounds round(s) to continue training.
    eval_metric: Union[str, Callbale]
      If string, it should be a built-in evaluation metric to use.
      If callable, it should be a custom evaluation metric, see official note for more details.
    """
    if model_dir is not None:
      if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
      model_dir = tempfile.mkdtemp()
    self.model_dir = model_dir
    self.model_instance = model_instance
    self.model_class = model_instance.__class__
    self.early_stopping_rounds = early_stopping_rounds
    self.model_type = self._check_model_type()

    if eval_metric is None:
      if self.model_type == 'classification':
        self.eval_metric: Union[str, Callable[..., Tuple]] = 'auc'
      elif self.model_type == 'regression':
        self.eval_metric = 'mae'
    else:
      self.eval_metric = eval_metric

  def _check_model_type(self):
    class_name = self.model_instance.__class__.__name__
    if class_name.endswith('Classifier'):
      return 'classification'
    elif class_name.endswith('Regressor'):
      return 'regression'
    else:
      raise ValueError(
          '{} is not a supported model instance.'.format(class_name))

  def fit(self, dataset: Dataset):
    """Fits GDBT model to data.

    First, this function splits all data into train and valid data (8:2),
    and finds the best n_estimators. And then, we retrain all data using
    best n_estimators * 1.25.

    Parameters
    ----------
    dataset: Dataset
      The `Dataset` to train this model on.
    """
    X = dataset.X
    y = np.squeeze(dataset.y)

    # GDBT doesn't support multi-output(task)
    if len(y.shape) == 2:
      raise ValueError("GDBT model doesn't support multi-output(task)")

    seed = self.model_instance.random_state
    stratify = None
    if self.model_type == 'classification':
      stratify = y

    # Find optimal n_estimators based on original learning_rate
    # and early_stopping_rounds
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=stratify)

    self.model_instance.fit(
        X_train,
        y_train,
        early_stopping_rounds=self.early_stopping_rounds,
        eval_metric=self.eval_metric,
        eval_set=[(X_test, y_test)])

    # retrain model to whole data using best n_estimators * 1.25
    estimated_best_round = np.round(self.model_instance.best_ntree_limit * 1.25)
    self.model_instance.n_estimators = np.int64(estimated_best_round)
    self.model_instance.fit(X, y, eval_metric=self.eval_metric)
