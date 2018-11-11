import datetime
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import multiprocessing

N_JOBS = multiprocessing.cpu_count()

CAT_COLS = ["term", "grade", "sub_grade", "emp_length", "home_ownership",
            "verification_status", "purpose",
            "initial_list_status", "application_type"]

def predict_and_write(clf, X_test, testids, n):
    x = clf.predict_proba(X_test)
    #x = x.max(1).reshape((-1, 1))
    a = testids.values.reshape((-1, 1))
    out = pd.DataFrame(np.column_stack((a,x)), columns=['id','prob1', 'prob2'])
    out.to_csv('mysubmission{}.txt'.format(n), index=False)

def single_data(testdf, traindf):
    testdf['loan_status'] = ''
    testids = testdf.id
    df = pd.concat([testdf, traindf], sort=False)
    testidx = df.id.isin(testids)
    trainidx = ~df.id.isin(testids)
    return df, testidx, trainidx, testids

def split_data(df, testidx, trainidx):
    X = df.loc[:, df.columns != 'loan_status'].values
    y = df['loan_status'].values
    X_train = X[trainidx]
    X_test = X[testidx]
    y_train = y[trainidx]
    return X_train, y_train, X_test

def process_data(df):
    df = df.drop(columns=["emp_title", "zip_code", "title", "addr_state"])
    df = pd.get_dummies(df, columns=CAT_COLS) #, drop_first=True)
    year = datetime.datetime.now().year
    df.earliest_cr_line = df.earliest_cr_line.apply(lambda x: year - int(x[-4:]))
    df['loan_status'] = np.where(df['loan_status']=="Fully Paid", 0, 1)
    df = df.fillna(df.mean())
    mu = df.mean()
    std = df.std()
    df = (df - mu) / std
    min_val = np.abs(df.min().min())
    df.loan_status = (df.loan_status * std.loan_status) + mu.loan_status
    return df

if __name__ == "__main__":
    testdf = pd.read_csv('test.csv')
    traindf = pd.read_csv('train.csv')
    df, testidx, trainidx, testids = single_data(testdf, traindf)
    df = process_data(df)
    X_train, y_train, X_test = split_data(df, testidx, trainidx)
    clf = LogisticRegression(random_state=0, solver='lbfgs',
                    multi_class='multinomial', n_jobs=N_JOBS).fit(X_train, y_train)
    predict_and_write(clf, X_test, testids, 1)
    clf = MLPClassifier(hidden_layer_sizes=(25,), alpha=1e-06, activation='relu',
                    max_iter=250).fit(X_train, y_train)
    predict_and_write(clf, X_test, testids, 2)
    clf = XGBClassifier(n_jobs=N_JOBS, n_estimators=250, booster='gbtree',  max_depth=3,
                    reg_alpha=0.001).fit(X_train, y_train)
    predict_and_write(clf, X_test, testids, 3)
