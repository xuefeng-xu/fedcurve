from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def _get_classifier(classifier):
    CLASSIFIERS = {
        "LogisticRegression": LogisticRegression,
        "SGDClassifier": lambda: SGDClassifier(loss="log_loss"),
        "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis,
        "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis,
        "SVC": lambda: SVC(probability=True),  # Ensure probability output
        "KNeighborsClassifier": KNeighborsClassifier,
        "RadiusNeighborsClassifier": RadiusNeighborsClassifier,
        "GaussianProcessClassifier": GaussianProcessClassifier,
        "BernoulliNB": BernoulliNB,
        "GaussianNB": GaussianNB,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "AdaBoostClassifier": AdaBoostClassifier,
        "BaggingClassifier": BaggingClassifier,
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "MLPClassifier": MLPClassifier,
        "XGBClassifier": XGBClassifier,
        "LGBMClassifier": lambda: LGBMClassifier(verbose=-1),
    }

    if classifier not in CLASSIFIERS:
        raise ValueError(f"Unknown classifier: {classifier}")

    clf = CLASSIFIERS[classifier]
    return clf() if callable(clf) else clf


def predict_proba(X, y, classifier):
    clf = _get_classifier(classifier)

    # Scikit-learn API
    clf.fit(X, y)
    return clf.predict_proba(X)[:, 1]
