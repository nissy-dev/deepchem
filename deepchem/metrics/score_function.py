"""Evaluation metrics."""

import numpy as np
from sklearn.metrics import matthews_corrcoef  # noqa
from sklearn.metrics import recall_score  # noqa
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import r2_score  # noqa
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score  # noqa
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score  # noqa
from sklearn.metrics import accuracy_score  # noqa
from sklearn.metrics import balanced_accuracy_score  # noqa
from scipy.stats import pearsonr

# kappa_score is an alias for `sklearn.metrics.cohen_kappa_score`
kappa_score = cohen_kappa_score


def pearson_r2_score(y: np.ndarray, y_pred: np.ndarray) -> float:
  """Computes Pearson R^2 (square of Pearson correlation).

  Parameters
  ----------
  y: np.ndarray
    ground truth array
  y_pred: np.ndarray
    predicted array

  Returns
  -------
  float
    The Pearson-R^2 score.
  """
  return pearsonr(y, y_pred)[0]**2


def jaccard_index(y: np.ndarray, y_pred: np.ndarray) -> float:
  """Computes Jaccard Index which is the Intersection Over Union metric
  which is commonly used in image segmentation tasks.

  DEPRECATED: WILL BE REMOVED IN A FUTURE VERSION OF DEEEPCHEM. USE `jaccard_score` instead.

  Parameters
  ----------
  y: np.ndarray
    ground truth array
  y_pred: np.ndarray
    predicted array

  Returns
  -------
  score: float
    The jaccard index. A number between 0 and 1.
  """
  return jaccard_score(y, y_pred)


def pixel_error(y: np.ndarray, y_pred: np.ndarray) -> float:
  """An error metric in case y, y_pred are images.

  Defined as 1 - the maximal F-score of pixel similarity, or squared
  Euclidean distance between the original and the result labels.

  Parameters
  ----------
  y: np.ndarray
    ground truth array
  y_pred: np.ndarray
    predicted array

  Returns
  -------
  score: float
    The pixel-error. A number between 0 and 1.
  """
  return 1 - f1_score(y, y_pred)


def prc_auc_score(y: np.ndarray, y_pred: np.ndarray) -> float:
  """Compute area under precision-recall curve

  Parameters
  ----------
  y: np.ndarray
    A numpy array of shape `(N, n_classes)` or `(N,)` with true labels
  y_pred: np.ndarray
    Of shape `(N, n_classes)` with class probabilities.

  Returns
  -------
  float
    The area under the precision-recall curve. A number between 0 and 1.
  """
  precision, recall, _ = precision_recall_curve(y[:, 1], y_pred[:, 1])
  return auc(recall, precision)


def rms_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Computes RMS error."""
  return np.sqrt(mean_squared_error(y_true, y_pred))


def mae_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Computes MAE."""
  return mean_absolute_error(y_true, y_pred)


def bedroc_score(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 20.0):
  """Compute BEDROC metric.

  BEDROC metric implemented according to Truchon and Bayley that modifies
  the ROC score by allowing for a factor of early recognition.
  Please confirm details from [1]_.

  Parameters
  ----------
  y_true: np.ndarray
    Binary class labels. 1 for positive class, 0 otherwise
  y_pred: np.ndarray
    Predicted labels
  alpha: float, default 20.0
    Early recognition parameter

  Returns
  -------
  float
    Value in [0, 1] that indicates the degree of early recognition

  Notes
  -----
  This function requires RDKit to be installed.

  References
  ----------
  .. [1] Truchon et al. "Evaluating virtual screening methods: good and bad metrics
     for the “early recognition” problem." Journal of chemical information and modeling
     47.2 (2007): 488-508.
  """
  try:
    from rdkit.ML.Scoring.Scoring import CalcBEDROC
  except ModuleNotFoundError:
    raise ImportError("This function requires RDKit to be installed.")

  # validation
  assert len(y_true) == len(y_pred), 'Number of examples do not match'
  assert np.array_equal(
      np.unique(y_true).astype(int),
      [0, 1]), ('Class labels must be binary: %s' % np.unique(y_true))

  yt = np.asarray(y_true)
  yp = np.asarray(y_pred)

  yt = yt.flatten()
  yp = yp[:, 1].flatten()  # Index 1 because one_hot predictions

  scores = list(zip(yt, yp))
  scores = sorted(scores, key=lambda pair: pair[1], reverse=True)

  return CalcBEDROC(scores, 0, alpha)
