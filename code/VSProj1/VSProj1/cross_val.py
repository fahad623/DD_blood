from sklearn.grid_search import GridSearchCV
from sklearn.metrics import log_loss, make_scorer

def cv_optimize(clf, X_train, Y_train, param_grid):
    #log_scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
    gs = GridSearchCV(clf, param_grid = param_grid, cv = 100, n_jobs = 7, verbose = 3)
    gs.fit(X_train, Y_train)
    print "gs.best_params_ = {0}, gs.best_score_ = {1}".format(gs.best_params_, gs.best_score_)
    return gs.best_estimator_, gs.best_params_, gs.best_score_


def fit_clf(clf, X_train, Y_train, param_grid = dict()):
    clf, bp, bs = cv_optimize(clf, X_train, Y_train, param_grid)    
    clf.fit(X_train, Y_train)
    return clf, bp, bs