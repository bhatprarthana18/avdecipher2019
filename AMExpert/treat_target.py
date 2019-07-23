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
    return 'my-error', mean_squared_log_error(labels,preds)

# Read the data
data = pd.read_csv("../Train/train.csv")
test_data = pd.read_csv("../test.csv")
## Aproach 1
# data.cc_cons[data.cc_cons>15763] = 15763
# data.cc_cons[data.cc_cons<560] = 560

## Aproach 2
data.cc_cons[data.cc_cons==0] = 1
data.cc_cons = np.log(data.cc_cons)
print(data.cc_cons.describe())

print(data.head())
print(test_data.head())
print(data.columns)

print(data.loan_enq.unique())
data['loan_enq'].fillna('N',inplace=True)
test_data['loan_enq'].fillna('N',inplace=True)
print(data.dtypes)

# Encode the categorical variables
le = preprocessing.LabelEncoder()
le.fit(data['account_type'])
print(le.classes_)
data['account_type'] = le.transform(data['account_type'])
test_data['account_type'] = le.transform(test_data['account_type'])

le1 = preprocessing.LabelEncoder()
le1.fit(data['gender'])
print(le1.classes_)
data['gender'] = le1.transform(data['gender'])
test_data['gender'] = le1.transform(test_data['gender'])

le2 = preprocessing.LabelEncoder()
le2.fit(data['loan_enq'])
print(le2.classes_)
data['loan_enq'] = le2.transform(data['loan_enq'])
test_data['loan_enq'] = le2.transform(test_data['loan_enq'])

print(data.dtypes)
#Build an xgboost model on raw data
y = data['cc_cons']
# X = data.iloc[,2:43]
X = data.iloc[:, 1:43]
print(X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

Y = test_data.iloc[:,1:43]
dfinal_test = xgb.DMatrix(Y)
# params = {'colsample_bylevel': 0.7, 'colsample_bytree': 0.7, 'learning_rate': 0.1,
#                       'max_depth': 10, 'min_child_weight': 1, 'n_estimators': 20,
#                       'objective': 'reg:linear', 'scale_pos_weight': 1, 'subsample': 1.0}



#
# best_xgb_model = XGBRegressor(n_estimators=10, learning_rate=0.08, gamma=0, subsample=0.75,
#                            colsample_bytree=1, max_depth=4,objective="reg:squarederror",verbosity = 0)

# best_xgb_model.fit(X_train,y_train)

########################################

# "Learn" the mean from the training data
mean_train = np.mean(y_train)
# Get predictions on the test set
baseline_predictions = np.ones(y_test.shape) * mean_train
# Compute MAE
mae_baseline = mean_absolute_error(y_test, baseline_predictions)
print("Baseline MSLE is {:.2f}".format(mae_baseline))

#########################################

# param = {'max_depth': 2, 'eta': 1, 'silent': 1}
# watchlist = [(dtest, 'eval'), (dtrain, 'train')]
# num_round = 2
# bst = xgb.train(param, dtrain, num_round, watchlist,
#                 feval="msle")

params = {
    # Parameters that we are going to tune.
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    # Other parameters
    'objective':'reg:squarederror'
}
num_boost_round = 100

model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=10,
    feval=msle
)
print("Best MSLE: {:.2f} with {} rounds".format(
                 model.best_score,
                 model.best_iteration+1))

# cv_results = xgb.cv(
#     params,
#     dtrain,
#     num_boost_round=num_boost_round,
#     seed=42,
#     nfold=5,
#     metrics={'mae'},
#     early_stopping_rounds=10
# )
#
# print(cv_results)
#
# print(cv_results['test-mae-mean'].min())
#

predictions = model.predict(dtest)
result= mean_squared_log_error(predictions,y_test)
print(result)


# Predict on final data
final_predictions = model.predict(dfinal_test)
print(final_predictions)
#Aproach 2
final_predictions = np.exp(final_predictions)
final_predictions[final_predictions==1] = 0
print(final_predictions)
final_submission_data = pd.DataFrame({'id':test_data.id,'cc_cons':final_predictions})
print(final_submission_data.head())


final_submission_data.to_csv("../Submissions/target_treated_1.csv",index=False)