import re
import csv
import codecs
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split, cross_val_predict
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from  sklearn.metrics import log_loss
import os.path
X, y = load_iris(return_X_y=True)
bind = y==0
y[bind] = 1
y[np.logical_not(bind)] = 0
ind_train, ind_test = train_test_split(np.arange(y.shape[0]), test_size = 0.3, random_state=42)
y_tr = y[ind_train]
y_out = y[ind_test]

#X_all = np.append(X_train, X_test, axis=0)
#train_mask = np.arange(X_all.shape[0]) < X_train.shape[0]
#test_mask = np.logical_not(train_mask)

# Как получить train, test из X_all
#[np.array_equal(X_all[train_mask], X_train),
# np.array_equal(X_all[test_mask], X_test)]

kf = KFold(random_state=42, n_splits=5, shuffle=True)

params = {'fit_intercept': False,
          'verbose': 1,
          'C':1.0
}
print("Define model...", params)
clf = LogisticRegression(**params)
#print("Fit...")
#clf.fit(X[ind_train], y[ind_train])

print("Predict...")
#y_pred = clf.predict_proba(X[ind_test])[:,1]
#print(log_loss(y_pred=y_pred, y_true=y[ind_test]))
#for pred, tr in zip(y_pred, y[ind_test]):
#    print(pred, tr)

#res = cross_val_predict(clf, X, y=y, cv=kf, n_jobs=4, verbose=True, method='predict_proba')[:,1]

def my_cross_val(clf, X, y):
    res = cross_val_predict(clf, X, y=y, cv=kf, verbose=True, method='predict_proba')[:,1]
    cv_error = [
        log_loss(y_pred=res[test_index], y_true=y[test_index])
                for train_index, test_index in kf.split(X)]
    print('log_loss =', log_loss(y_pred=res, y_true=y),
           '\nlog_loss for each fold mean and sd =', [np.mean(cv_error), np.std(cv_error)])
    print(clf)
    print(clf.get_params())
    return res, np.mean(cv_error), np.std(cv_error)

STACKING_FOLDER = "../models_stacking/"
file_model_df = STACKING_FOLDER + "models_base.csv"
if not(os.path.isfile(file_model_df)):
    df = pd.DataFrame(columns=['clf', 'params', 'type', 'log_loss_all', 'log_loss_mean', 'log_loss_sd'])
    df.to_csv(file_model_df, index=False)
df = pd.read_csv(file_model_df)

l = []
params = {'fit_intercept': False,
          'verbose': 0,
          'C':1.0}
clf = LogisticRegression(**params)
l.append((clf, params))

params = {'fit_intercept': False,
          'verbose': 0,
          'C':0.5}
clf = LogisticRegression(**params)
l.append((clf, params))


params = {'fit_intercept': False,
          'verbose': 0,
          'C':0.2}
clf = LogisticRegression(**params)
l.append((clf, params))


for clf, params in l:
    res, error_mean, error_sd = my_cross_val(clf, X[ind_train], y[ind_train])
    clf.fit(X[ind_train], y[ind_train])
    res_test = clf.predict_proba(X[ind_test])[:,1]

    df = df.append({'clf': 'LogReg', 'params': params,
                    'type': type(clf),
                    'log_loss_mean': error_mean,
                    'log_loss_sd': error_sd}, ignore_index=True)
    df.to_csv(file_model_df, index=False)
    t = df.index[-1], df.iloc[df.index[-1], 0]
    res_file = '_'.join([str(x) for x in t])
    res_file_train = STACKING_FOLDER + res_file + '_train.csv'
    res_file_test = STACKING_FOLDER + res_file + '_test.csv'


    ## write files
    for r, file in [(res, res_file_train), (res_test, res_file_test)]:
        with open(file, 'wb') as abc:
            np.savetxt(abc, r, delimiter=",")
        print(r.shape, file)

"""
for clf, X in clf_feat:
    res = my_cross_val(clf, X[ind_train], y_tr)
    clf.fit(X[ind_train], y_tr)
    y_out_pred = clf.predict_proba(X[ind_test])[:, 1];
    print(log_loss(y_out, y_out_pred))

    y_tr_prob_array[:, i] = res
    y_out_prob_array[:, i] = y_out_pred
    i = i + 1




"""