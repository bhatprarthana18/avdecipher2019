import pandas as pd


import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
# from xgboost.sklearn import XGBRegressor
from sklearn import preprocessing
import xgboost as xgb
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error

def msle(preds, dtrain):
    labels = dtrain.get_label()
    # return a pair metric_name, result. The metric name must not contain a
    # colon (:) or a space since preds are margin(before logistic
    # transformation, cutoff at 0)
    return 'MSLE', mean_squared_log_error(labels,preds)

# Read the data
train_data = pd.read_csv("../new_train_data.csv")
test_data = pd.read_csv("../new_test_data.csv")

# Encode the categorical variables
le = preprocessing.LabelEncoder()
le.fit(train_data['account_type'])
print(le.classes_)
train_data['account_type'] = le.transform(train_data['account_type'])
test_data['account_type'] = le.transform(test_data['account_type'])

le1 = preprocessing.LabelEncoder()
le1.fit(train_data['gender'])
print(le1.classes_)
train_data['gender'] = le1.transform(train_data['gender'])
test_data['gender'] = le1.transform(test_data['gender'])

le2 = preprocessing.LabelEncoder()
le2.fit(train_data['loan_enq'])
print(le2.classes_)
train_data['loan_enq'] = le2.transform(train_data['loan_enq'])
test_data['loan_enq'] = le2.transform(test_data['loan_enq'])


id = "id"
target = 'cc_cons'
train_data.cc_cons[train_data.cc_cons==0] = 1
train_data['cc_cons'] = np.log(train_data['cc_cons'])
print(train_data[target].describe())

all_cols = list(train_data.columns)

predictors = [x for x in all_cols if x not in [id,target]]
print(predictors)

y = train_data[target]
X = train_data[predictors]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
Y = test_data[predictors]
dfinal_test = xgb.DMatrix(Y)


#################
final_dtrain = xgb.DMatrix(X, label=y)

watchlist = [(dtest, 'eval'), (dtrain, 'train')]

params = {
    # Parameters that we are going to tune.
    'max_depth':9,
    'min_child_weight': 8,
    'eta':.1,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    # Other parameters
    'objective':'reg:squarederror',
    "booster":"gbtree"

}
# num_boost_round = 500
# cv_results = xgb.cv(
#     params,
#     dtrain,
#     num_boost_round=num_boost_round,
#     seed=42,
#     nfold=5,
#     metrics={'mae'},
#     early_stopping_rounds=10)
# print(cv_results)
# print(cv_results['test-mae-mean'].min())

model = xgb.train(
    params,
    final_dtrain,
    num_boost_round = 75,
    feval=msle
)

# print("Best MSLE: {:.2f} with {} rounds".format(
#                  model.best_score,
#                  model.best_iteration+1))





# predictions = model.predict(dtest)
# result= mean_squared_log_error(predictions,y_test)
# print(result)
#
#
# Predict on final data
final_predictions = model.predict(dfinal_test)
print(final_predictions)
final_predictions = np.exp(final_predictions)
final_submission_data = pd.DataFrame({'id':test_data.id,'cc_cons':final_predictions})
print(final_submission_data.head())
final_submission_data.to_csv("../Submissions/final_model.csv",index=False)