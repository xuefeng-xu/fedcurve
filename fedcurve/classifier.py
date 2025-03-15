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


def get_classifier(classifier):
    if classifier == "LogisticRegression":
        clf = LogisticRegression()
    elif classifier == "SGDClassifier":
        clf = SGDClassifier(loss="log_loss")
    elif classifier == "LinearDiscriminantAnalysis":
        clf = LinearDiscriminantAnalysis()
    elif classifier == "QuadraticDiscriminantAnalysis":
        clf = QuadraticDiscriminantAnalysis()
    elif classifier == "SVC":
        clf = SVC(probability=True)
    elif classifier == "KNeighborsClassifier":
        clf = KNeighborsClassifier()
    elif classifier == "RadiusNeighborsClassifier":
        clf = RadiusNeighborsClassifier()
    elif classifier == "GaussianProcessClassifier":
        clf = GaussianProcessClassifier()
    elif classifier == "BernoulliNB":
        clf = BernoulliNB()
    elif classifier == "GaussianNB":
        clf = GaussianNB()
    elif classifier == "DecisionTreeClassifier":
        clf = DecisionTreeClassifier()
    elif classifier == "AdaBoostClassifier":
        clf = AdaBoostClassifier()
    elif classifier == "BaggingClassifier":
        clf = BaggingClassifier()
    elif classifier == "ExtraTreesClassifier":
        clf = ExtraTreesClassifier()
    elif classifier == "GradientBoostingClassifier":
        clf = GradientBoostingClassifier()
    elif classifier == "HistGradientBoostingClassifier":
        clf = HistGradientBoostingClassifier()
    elif classifier == "RandomForestClassifier":
        clf = RandomForestClassifier()
    elif classifier == "MLPClassifier":
        clf = MLPClassifier()
    elif classifier == "XGBClassifier":
        clf = XGBClassifier()
    elif classifier == "LGBMClassifier":
        clf = LGBMClassifier(verbose=-1)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    return clf


def predict_proba(X, y, classifier):
    clf = get_classifier(classifier)

    # Scikit-learn API
    clf.fit(X, y)
    return clf.predict_proba(X)[:, 1]
