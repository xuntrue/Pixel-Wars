import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def logistic(fit_intercept: bool) -> Pipeline:
    """
    Ridge logistic regression with the regularisation strength tuned by an inner CV.
    With fit_intercept=False and A-minus-B features, equal armies predict exactly
    50%, and swapping sides flips the prediction (so side-swap augmentation would
    be redundant). The scaler must not centre the data, which would break that symmetry.
    """
    return Pipeline([
        ("scale", StandardScaler(with_mean=False)),
        ("clf", LogisticRegressionCV(
            Cs=np.logspace(-3, 2, 12),
            cv=StratifiedKFold(5, shuffle=True, random_state=0),
            scoring="neg_log_loss",
            fit_intercept=fit_intercept,
            max_iter=2000,
        )),
    ])


# (name, model factory or None, feature set or None)
CANDIDATES = [
    ("coin flip (0.5)", None, None),
    ("base rate (A win share)", lambda: DummyClassifier(strategy="prior"), "hp_only"),
    ("logreg [hp_only]", lambda: logistic(False), "hp_only"),
    ("logreg [size_only]", lambda: logistic(False), "size_only"),
    ("logreg [core]", lambda: logistic(False), "core"),
    ("logreg [core+rates]", lambda: logistic(False), "core+rates"),
    ("logreg [all]", lambda: logistic(False), "all"),
    ("logreg+intercept [all]", lambda: logistic(True), "all"),
]
BASELINES = {"coin flip (0.5)", "base rate (A win share)"}
REFERENCE = "base rate (A win share)"  # models are compared against this one