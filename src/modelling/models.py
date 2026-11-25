import numpy as np
import pandas as pd
import time

from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier # to be the baseline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.analysis.stats import line, note, show, subtitle, title

from .features import mirror_X

VERBOSE = True    # show training progress in the terminal

# --- Helper funcion(s) for SymmetricClassifier and oof_predictions() ---
def _dot() -> None:
    """ One dot per fitted base model, on the same line as the fold message """
    if VERBOSE:
        print(".", end="", flush=True)

def _score(model, X) -> np.ndarray:
    """ Log-odds that A wins: from predict_proba if the model has it, else its decision_function """
    if hasattr(model, "predict_proba"):
        p = np.clip(model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
    return model.decision_function(X)

def _group_ids(X: pd.DataFrame) -> np.ndarray:
    """ Same key for a battle and its mirror image, so repeats stay in one inner fold """
    h = pd.util.hash_pandas_object(X, index=False).to_numpy()
    hm = pd.util.hash_pandas_object(mirror_X(X), index=False).to_numpy()
    return np.minimum(h, hm)

def _fit_base(self, X, y):
    Xa = pd.concat([X, mirror_X(X)], ignore_index=True)
    ya = np.concatenate([y, 1 - y])
    model = clone(self.estimator).fit(Xa, ya)
    _dot()
    return model

class SymmetricClassifier(ClassifierMixin, BaseEstimator):
    """
    Makes any classifier treat armies A and B interchangeably.
      fit:     trains on the battles plus their mirror images (A and B swapped, label flipped)
      predict: averages the score over both orderings, so P(A beats B) = 1 - P(B beats A)
      calibrate=True: fits one temperature t so P = sigmoid(t * score), on out-of-fold scores.
    X must be a DataFrame whose columns follow the A_/B_ naming that mirror_X understands.
    """
    def __init__(self, estimator, calibrate: bool = True, cal_splits: int = 5):
        self.estimator = estimator
        self.calibrate = calibrate
        self.cal_splits = cal_splits

    def _fit_base(self, X, y):
        Xa = pd.concat([X, mirror_X(X)], ignore_index=True)
        ya = np.concatenate([y, 1 - y])
        return clone(self.estimator).fit(Xa, ya)

    def _sym_score(self, model, X) -> np.ndarray:
        return 0.5 * (_score(model, X) - _score(model, mirror_X(X)))

    def fit(self, X, y):
        y = np.asarray(y)
        self.estimator_ = self._fit_base(X, y)
        self.t_ = 1.0
        if self.calibrate:
            oof = np.zeros(len(y))
            cv = StratifiedGroupKFold(self.cal_splits, shuffle=True, random_state=0)
            for tr, te in cv.split(X, y, _group_ids(X)):
                m = self._fit_base(X.iloc[tr], y[tr])      # mirror made AFTER the split: no leak
                oof[te] = self._sym_score(m, X.iloc[te])
            self.t_ = float(LogisticRegression(C=1e6, fit_intercept=False)
                            .fit(oof.reshape(-1, 1), y).coef_[0, 0])
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = expit(self.t_ * self._sym_score(self.estimator_, X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

def logistic(fit_intercept: bool, wide: bool = False) -> Pipeline:
    """
    Ridge logistic regression with the regularisation strength tuned by an inner CV.
    With fit_intercept=False and A-minus-B features, equal armies predict exactly 50%,
    and swapping sides flips the prediction (so side-swap augmentation would be redundant).
    The scaler must not centre the data, which would break that symmetry.
    wide=True searches stronger regularisation, for feature sets with hundreds of columns.
    """
    Cs = np.logspace(-5, 0, 12) if wide else np.logspace(-3, 2, 12)
    return Pipeline([
        ("scale", StandardScaler(with_mean=False)),
        ("clf", LogisticRegressionCV(
            Cs=Cs,
            cv=StratifiedKFold(5, shuffle=True, random_state=0),
            scoring="neg_log_loss",
            fit_intercept=fit_intercept,
            max_iter=2000,
            l1_ratios=(0,),               # pure L2 (ridge), stated explicitly
            use_legacy_attributes=True,   # keep clf.C_[0] and clf.coef_[0] as train.py uses them
        )),
    ])

def random_forest():
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=0
    )

def grad_boost():
    return HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.05,
        max_iter=150,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,      # its internal validation split would contain mirrors of training rows
        # more HP, attack or units for A must never lower A's win chance (game knowledge, curbs overfitting)
        monotonic_cst={"log_hp_ratio": 1, "log_atk_ratio": 1, "log_size_ratio": 1},
        random_state=0
    )

def svm():
    # RBF SVM: only a decision score, so it relies on the wrapper's calibration for probabilities
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale"))
    ])

def symmetric(estimator) -> SymmetricClassifier:
    return SymmetricClassifier(estimator, calibrate=True)

# (name, model factory or None, feature set or None)
# Set names are resolved by train.py via feature_sets().
CANDIDATES = [
    # ("coin flip (0.5)", None, None),
    ("base rate (A win share)", lambda: DummyClassifier(strategy="prior"), "hp_only"),

    # Logistic Regression, aggregate features
    ("logreg [hp_only]", lambda: logistic(False), "hp_only"),
    ("logreg [size_only]", lambda: logistic(False), "size_only"),
    ("logreg [core]", lambda: logistic(False), "core"),
    ("logreg [core+rates]", lambda: logistic(False), "core+rates"),
    ("logreg [all]", lambda: logistic(False), "all"),
    ("logreg+intercept [all]", lambda: logistic(True), "all"),

    # Logistic Regression, army composition (antisymmetric, so still symmetric by construction)
    ("logreg [units+arts]", lambda: logistic(False), "units+arts"),
    ("logreg [unit_matchups]", lambda: logistic(False, wide=True), "unit_matchups"),
    ("logreg [full_matchups]", lambda: logistic(False, wide=True), "full_matchups"),

    # Nonlinear models on aggregate features only
    #("random forest [all]", lambda: symmetric(random_forest()), "all"),
    #("grad boost [all]", lambda: symmetric(grad_boost()), "all"),
    #("svm [all]", lambda: symmetric(svm()), "all"),

    # Nonlinear models with composition: can they find unit and artifact matchups?
    #("random forest [trees_raw]", lambda: symmetric(random_forest()), "trees_raw"),
    #("grad boost [trees_raw]", lambda: symmetric(grad_boost()), "trees_raw"),
    #("svm [units+arts]", lambda: symmetric(svm()), "units+arts")
]
BASELINES = {"coin flip (0.5)", "base rate (A win share)"}
REFERENCE = "base rate (A win share)"   # models are compared against this one
CHAMPION = "logreg [all]"               # the model any new candidate has to beat