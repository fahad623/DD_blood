from sklearn.grid_search import GridSearchCV
from sklearn.metrics import log_loss, make_scorer

def make_decision_scorer():
    return make_scorer(log_loss, greater_is_better = False, needs_threshold = True)

def cv_optimize(clf, X_train, Y_train, param_grid, scorer):
    gs = GridSearchCV(clf, param_grid = param_grid, scoring = scorer, cv = 100, n_jobs = 7, verbose = 3)
    gs.fit(X_train, Y_train)
    print "gs.best_params_ = {0}, gs.best_score_ = {1}".format(gs.best_params_, gs.best_score_)
    print "gs.grid_scores_ = {0}".format(gs.grid_scores_)
    return gs.best_estimator_, gs.best_params_, gs.best_score_


def fit_clf(clf, X_train, Y_train, param_grid = dict(), scorer = None):
    clf, bp, bs = cv_optimize(clf, X_train, Y_train, param_grid, scorer)    
    clf.fit(X_train, Y_train)
    return clf, bp, bs