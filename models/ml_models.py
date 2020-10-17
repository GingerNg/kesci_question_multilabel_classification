
from sklearn.svm import LinearSVC
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB


def svm():
    svc = LinearSVC(loss='hinge', dual=True)
    param_distributions = {'C': uniform(0, 10)}
    model = RandomizedSearchCV(
        estimator=svc, param_distributions=param_distributions, cv=3, n_iter=50, verbose=1)
    return model


def bayes():
    nb = MultinomialNB()
    return nb
