import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

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

def create_bins(data,col,bins,labels):
    data[col].fillna(0, inplace=True)
    data[col+'_bin'] = pd.cut(data[col],
                 bins=bins,
                 labels=labels)

    print(data[[col,col+'_bin']].head())
    print(data[[col,col+'_bin']].sort_values(by = [col],ascending=False).head())

    return data


# Read the train and test data
train_data = pd.read_csv("../Train/train.csv")
test_data = pd.read_csv("../test.csv")
train_data['tag_col'] = "Train"
test_data["tag_col"] = "Test"
merged_data = pd.concat([train_data,test_data],sort=False)

#Target
print(merged_data.cc_cons.describe())
# print(np.percentile(data.cc_cons,90))
# print(np.percentile(data.cc_cons,10))

# data.cc_cons[data.cc_cons>15763] = 15763
# data.cc_cons[data.cc_cons<560] = 560
# Take Log instead


#.................................................
#Age
merged_data[merged_data.age>100]['age'] = 30
print(merged_data.age.describe())
#.................................................


#.................................................
#Check if same regions are there in train and test

# Group the regions
merged_data = create_bins(merged_data,
            'region_code',
            bins=[100,200,300,400,500,600,700,800,900,1000],
            labels=[200,300,400,500,600,700,800,900,1000])
#.................................................
#Group the expenses of creadit card

bins_x = [-1,100,1000,3000,5000,10000,25000,50000,2000000]
labels_x = [100,1000,3000,5000,10000,25000,50000,2000000]

merged_data = create_bins(merged_data,'cc_cons_apr',bins_x,labels_x)
merged_data = create_bins(merged_data,'cc_cons_may',bins_x,labels_x)
merged_data = create_bins(merged_data,'cc_cons_jun',bins_x,labels_x)
merged_data = create_bins(merged_data,'dc_cons_apr',bins_x,labels_x)
merged_data = create_bins(merged_data,'dc_cons_may',bins_x,labels_x)
merged_data = create_bins(merged_data,'dc_cons_jun',bins_x,labels_x)


print(merged_data[['cc_cons_jun','cc_cons_jun_bin']].head())
print(merged_data[['cc_cons_apr','cc_cons_apr_bin']].head())
print(merged_data[['cc_cons_may','cc_cons_may_bin']].head())

print(merged_data[['cc_cons_jun','cc_cons_jun_bin']].sort_values(by = ['cc_cons_jun'],ascending=False).head())
print(merged_data[['cc_cons_may','cc_cons_may_bin']].sort_values(by = ['cc_cons_may'],ascending=False).head())
print(merged_data[['cc_cons_apr','cc_cons_apr_bin']].sort_values(by = ['cc_cons_apr'],ascending=False).head())

print(merged_data[['dc_cons_jun','dc_cons_jun_bin']].head())
print(merged_data[['dc_cons_apr','dc_cons_apr_bin']].head())
print(merged_data[['dc_cons_may','dc_cons_may_bin']].head())

print(merged_data[['dc_cons_jun','dc_cons_jun_bin']].sort_values(by = ['dc_cons_jun'],ascending=False).head())
print(merged_data[['dc_cons_may','dc_cons_may_bin']].sort_values(by = ['dc_cons_may'],ascending=False).head())
print(merged_data[['dc_cons_apr','dc_cons_apr_bin']].sort_values(by = ['dc_cons_apr'],ascending=False).head())

# .................................................
# count Debit card transactions in last three months
bins_x = [-1, 0, 1, 5, 20, 50, 100, 5000]
labels_x = [0, 1, 5, 20, 50, 100, 5000]
merged_data = create_bins(merged_data,'cc_count_apr',bins_x,labels_x)
merged_data = create_bins(merged_data,'cc_count_may',bins_x,labels_x)
merged_data = create_bins(merged_data,'cc_count_jun',bins_x,labels_x)
merged_data = create_bins(merged_data,'dc_count_apr',bins_x,labels_x)
merged_data = create_bins(merged_data,'dc_count_may',bins_x,labels_x)
merged_data = create_bins(merged_data,'dc_count_jun',bins_x,labels_x)

#.................................................
# Card Limits
merged_data['card_lim'].fillna(10000,inplace=True)
merged_data.loc[merged_data['card_lim'] <10000, 'card_lim'] = 10000
print(merged_data[['id','card_lim']].sort_values(by = ['card_lim'],ascending=False).head())
print(merged_data[['id','card_lim']].sort_values(by = ['card_lim'],ascending=True).head())

#.................................................
# Personal Loan Active
merged_data['personal_loan_active'].fillna(0,inplace=True)
merged_data['personal_loan_closed'].fillna(0,inplace=True)
merged_data['vehicle_loan_active'].fillna(0,inplace=True)
merged_data['vehicle_loan_closed'].fillna(0,inplace=True)

print(merged_data[['personal_loan_active','personal_loan_closed','vehicle_loan_active','vehicle_loan_closed']].head())
#.................................................

# Investments
merged_data['investment_1'].fillna(0,inplace=True)
merged_data['investment_2'].fillna(0,inplace=True)
merged_data['investment_3'].fillna(0,inplace=True)
merged_data['investment_4'].fillna(0,inplace=True)

#.............................................
# Loan Enq
merged_data['loan_enq'].fillna('N',inplace=True)
#.............................................

# Credit amount apr
bins_x = [-1,0,15000,20000,30000,40000,55000,70000,95000,150000,50000000]
labels_x = [0,15000,20000,30000,40000,55000,70000,95000,150000,50000000]
merged_data = create_bins(merged_data,col='credit_amount_apr',bins=bins_x,labels=labels_x)

bins_x = [-1,0,10000,20000,30000,40000,55000,70000,95000,150000,50000000]
labels_x = [0,10000,20000,30000,40000,55000,70000,95000,150000,50000000]
merged_data = create_bins(merged_data,col='credit_amount_jun',bins=bins_x,labels=labels_x)

bins_x = [-1,0,10000,20000,30000,40000,55000,70000,95000,150000,50000000]
labels_x = [0,10000,20000,30000,40000,55000,70000,95000,150000,50000000]
merged_data = create_bins(merged_data,col='credit_amount_may',bins=bins_x,labels=labels_x)

bins_x = [-1,0,15000,20000,30000,40000,55000,70000,95000,150000,50000000]
labels_x = [0,15000,20000,30000,40000,55000,70000,95000,150000,50000000]
merged_data = create_bins(merged_data,col='debit_amount_apr',bins=bins_x,labels=labels_x)

bins_x = [-1,0,10000,20000,30000,40000,55000,70000,95000,150000,50000000]
labels_x = [0,10000,20000,30000,40000,55000,70000,95000,150000,50000000]
merged_data = create_bins(merged_data,col='debit_amount_jun',bins=bins_x,labels=labels_x)

bins_x = [-1,0,10000,20000,30000,40000,55000,70000,95000,150000,50000000]
labels_x = [0,10000,20000,30000,40000,55000,70000,95000,150000,50000000]
merged_data = create_bins(merged_data,col='debit_amount_may',bins=bins_x,labels=labels_x)

# Max credt amount
bins_x = [-1,0,15000,20000,30000,40000,55000,70000,95000,150000,50000000]
labels_x = [0,15000,20000,30000,40000,55000,70000,95000,150000,50000000]
merged_data = create_bins(merged_data,col='max_credit_amount_apr',bins=bins_x,labels=labels_x)

bins_x = [-1,0,10000,20000,30000,40000,55000,70000,95000,150000,50000000]
labels_x = [0,10000,20000,30000,40000,55000,70000,95000,150000,50000000]
merged_data = create_bins(merged_data,col='max_credit_amount_jun',bins=bins_x,labels=labels_x)

bins_x = [-1,0,10000,20000,30000,40000,55000,70000,95000,150000,50000000]
labels_x = [0,10000,20000,30000,40000,55000,70000,95000,150000,50000000]
merged_data = create_bins(merged_data,col='max_credit_amount_may',bins=bins_x,labels=labels_x)

#/........................
# Credit and Debit Count
bins_x = [-1,0,1,2,4,6,20,500]
labels_x = [0,1,2,4,6,20,500]
merged_data = create_bins(merged_data,col='credit_count_may',bins=bins_x,labels=labels_x)
merged_data = create_bins(merged_data,col='credit_count_jun',bins=bins_x,labels=labels_x)
merged_data = create_bins(merged_data,col='credit_count_apr',bins=bins_x,labels=labels_x)
merged_data = create_bins(merged_data,col='debit_count_may',bins=bins_x,labels=labels_x)
merged_data = create_bins(merged_data,col='debit_count_apr',bins=bins_x,labels=labels_x)
merged_data = create_bins(merged_data,col='debit_count_jun',bins=bins_x,labels=labels_x)
#################################################################################################

print(merged_data.columns)
print(merged_data.dtypes)


#One hot encoding
column_to_be_encoded=['region_code_bin', 'cc_cons_apr_bin', 'cc_cons_may_bin',
       'cc_cons_jun_bin', 'dc_cons_apr_bin', 'dc_cons_may_bin',
       'dc_cons_jun_bin', 'cc_count_apr_bin', 'cc_count_may_bin',
       'cc_count_jun_bin', 'dc_count_apr_bin', 'dc_count_may_bin',
       'dc_count_jun_bin', 'credit_amount_apr_bin', 'credit_amount_jun_bin',
       'credit_amount_may_bin','debit_amount_apr_bin', 'debit_amount_jun_bin',
       'debit_amount_may_bin', 'max_credit_amount_apr_bin',
       'max_credit_amount_jun_bin', 'max_credit_amount_may_bin',
       'credit_count_may_bin', 'credit_count_jun_bin', 'credit_count_apr_bin',
       'debit_count_may_bin', 'debit_count_apr_bin', 'debit_count_jun_bin']

merged_data[column_to_be_encoded] = merged_data[column_to_be_encoded].astype('object')
# Get one hot encoding of columns B
one_hot = pd.get_dummies(merged_data[column_to_be_encoded])
# print(one_hot)
# Drop column B as it is now encoded
merged_data = merged_data.drop(column_to_be_encoded,axis = 1)
# Join the encoded df
merged_data = merged_data.join(one_hot)

print(merged_data.columns)
print(merged_data.dtypes)
print(merged_data.columns[merged_data.isna().any()].tolist())

# Split the train and test data and save it in files
train_data = merged_data[merged_data.tag_col=="Train"]
test_data = merged_data[merged_data.tag_col=="Test"]
train_data = train_data.drop(['tag_col'],axis=1)
test_data = test_data.drop(['tag_col'],axis=1)
train_data.to_csv("../new_train_data.csv",index=False)
test_data.to_csv("../new_test_data.csv",index=False)

