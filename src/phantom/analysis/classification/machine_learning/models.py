"""
The classifiers with their default hyperparameters, wrapped to apply SMOTE
oversampling inside each training fold only; and the late-fusion model
combining one classifier per modality.

``MODELS`` is the one list of algorithms every stage offers:

    rf        random forest (300 shallow trees)
    xgb       XGBoost (gradient-boosted trees)
    catboost  CatBoost (ordered boosting)
    logreg    L2 logistic regression on standardized features
    hgb       scikit-learn's histogram gradient boosting

All are class-weighted for imbalance where the algorithm supports it, and
take the job's ``seed``.
"""

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


class SingleModalModel(BaseEstimator, ClassifierMixin):
    def __init__(self, model, use_smote=False, smote_k=2, random_state=42):
        self.model = model
        self.use_smote = use_smote
        self.smote_k = smote_k
        self.random_state = random_state
        self.model_ = None

    def fit(self, X, y):
        self.model_ = clone(self.model)
        X_train, y_train = X, y
        if self.use_smote:
            k = min(self.smote_k, np.min(np.bincount(y)) - 1)
            if k > 0:
                smote = SMOTE(random_state=self.random_state, k_neighbors=k)
                X_train, y_train = smote.fit_resample(X, y)
        self.model_.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    @property
    def feature_importances_(self):
        """The model's built-in importances: impurity/gain for the tree models,
        |coefficient| (on standardized features) for logistic regression; NaN
        for a model without any (histogram gradient boosting)."""
        if self.model_ is None:
            return np.zeros(1)
        model = self.model_.steps[-1][1] if isinstance(self.model_, Pipeline) else self.model_
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_
        if hasattr(model, "get_feature_importance"):
            return model.get_feature_importance()
        if hasattr(model, "coef_"):
            return np.abs(np.asarray(model.coef_)).ravel()
        return np.full(model.n_features_in_, np.nan)


class MultiModalModel(BaseEstimator, ClassifierMixin):
    def __init__(self, models_dict=None, weights=None):
        self.models_dict = models_dict if models_dict is not None else {}
        self.weights = weights if weights is not None else {}

    def fit(self, X_blocks_dict, y):
        for modality, X in X_blocks_dict.items():
            if modality in self.models_dict:
                self.models_dict[modality].fit(X, y)
        return self

    def predict_proba(self, X_blocks_dict):
        probs = []
        modality_weights = []
        for modality, X in X_blocks_dict.items():
            if modality in self.models_dict:
                prob = self.models_dict[modality].predict_proba(X)[:, 1]
                probs.append(prob)
                weight = self.weights.get(modality, 1.0)
                modality_weights.append(weight)
        avg_prob = np.average(probs, axis=0, weights=modality_weights)
        return np.column_stack([1 - avg_prob, avg_prob])

    def predict(self, X_blocks_dict):
        return (self.predict_proba(X_blocks_dict)[:, 1] >= 0.5).astype(int)


def get_rf_model(use_smote=False, seed=42, **params):
    default_params = {
        "n_estimators": 300,
        "max_depth": 3,
        "min_samples_split": 4,
        "min_samples_leaf": 3,
        "max_features": "sqrt",
        "bootstrap": True,
        "class_weight": "balanced",
        "random_state": seed,
    }
    default_params.update(params)
    return SingleModalModel(
        RandomForestClassifier(**default_params), use_smote=use_smote, random_state=seed
    )


def get_xgb_model(use_smote=False, seed=42, **params):
    from xgboost import XGBClassifier

    default_params = {
        "n_estimators": 300,
        "max_depth": 2,
        "learning_rate": 0.03,
        "min_child_weight": 8,
        "gamma": 1.0,
        "subsample": 0.7,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.5,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "random_state": seed,
    }
    default_params.update(params)
    return SingleModalModel(XGBClassifier(**default_params), use_smote=use_smote, random_state=seed)


def get_catboost_model(use_smote=False, seed=42, **params):
    from catboost import CatBoostClassifier

    default_params = {
        "iterations": 50,
        "depth": 3,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5,
        "random_strength": 1,
        "bagging_temperature": 1,
        "auto_class_weights": "Balanced",
        "loss_function": "Logloss",
        "eval_metric": "F1",
        "verbose": 0,
        "random_state": seed,
        "allow_writing_files": False,  # no catboost_info/ training logs in the cwd
    }
    default_params.update(params)
    return SingleModalModel(
        CatBoostClassifier(**default_params), use_smote=use_smote, random_state=seed
    )


def get_logreg_model(use_smote=False, seed=42, **params):
    default_params = {
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 5000,
        "random_state": seed,
    }
    default_params.update(params)
    return SingleModalModel(
        make_pipeline(StandardScaler(), LogisticRegression(**default_params)),
        use_smote=use_smote,
        random_state=seed,
    )


def get_hgb_model(use_smote=False, seed=42, **params):
    default_params = {
        "learning_rate": 0.05,
        "max_iter": 200,
        "max_depth": 3,
        "min_samples_leaf": 5,
        "l2_regularization": 1.0,
        "class_weight": "balanced",
        # A random validation split of a few dozen samples is noise, not a stopping rule.
        "early_stopping": False,
        "random_state": seed,
    }
    default_params.update(params)
    return SingleModalModel(
        HistGradientBoostingClassifier(**default_params), use_smote=use_smote, random_state=seed
    )


# Every algorithm a stage can choose: name -> factory(use_smote=..., seed=..., **params).
MODELS = {
    "rf": get_rf_model,
    "xgb": get_xgb_model,
    "catboost": get_catboost_model,
    "logreg": get_logreg_model,
    "hgb": get_hgb_model,
}
# Models without built-in feature importances (their LOOCV importances are NaN;
# permutation importances under repeated CV are available).
NO_BUILTIN_IMPORTANCE = frozenset({"hgb"})
