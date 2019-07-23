import pandas as pd
from matplotlib import pyplot as plt
import numpy as np



# merged_data['emi_active'].fillna(0,inplace=True)
# res,bins = pd.qcut(merged_data['emi_active'],4,retbins=True)
# print(bins)
# print(res)

def create_bins(data,col,bins,labels):
    data[col].fillna(0, inplace=True)
    data[col+'_bin'] = pd.cut(data[col],
                 bins=bins,
                 labels=labels)

    print(data[[col,col+'_bin']].head())
    print(data[[col,col+'_bin']].sort_values(by = [col],ascending=False).head())

    return data


# Read the train and test data
data = pd.read_csv("../Train/train.csv")
test_data = pd.read_csv("../test.csv")

#.................................................
# Remove age > 71
# print(len(data[data.age>71]))
# print(len(test_data[test_data.age>71]))

# print(test_data.age.unique())
# plt.hist(data.age, bins=40)
# plt.ylabel('Age Bins')
# plt.show()

data[data.age>100] = 30
test_data[test_data.age>100] = 30

print(data.age.describe())
print(test_data.age.describe())
#.................................................


# # plt.hist(data.cc_cons, bins=40)
# # plt.ylabel('Age Bins')
# # plt.show()
#
# # res = pd.qcut(data.cc_cons_apr,10)
# # print(res)
#
# print(data.cc_cons.describe())
# # print(np.percentile(data.cc_cons,90))
# # print(np.percentile(data.cc_cons,10))
#
# # data.cc_cons[data.cc_cons>15763] = 15763
# # data.cc_cons[data.cc_cons<560] = 560
#
# # plt.hist(data.cc_cons, bins=40)
# # plt.ylabel('Age Bins')
# # plt.show()
# #.................................................
# #Take Log instead
#
#
# #.................................................
# #Check if same regions are there in train and test
# train_regions = data.region_code.unique()
# test_regions = test_data.region_code.unique()
# new_regions =[x for x in test_regions if x not in train_regions]
# print(new_regions)
#
# # Group the regions
# data['region_code_binned'] = pd.cut(data.region_code,bins=[100,200,300,400,500,600,700,800,900,1000],labels=[200,300,400,500,600,700,800,900,1000])
# test_data['region_code_binned'] = pd.cut(test_data.region_code,bins=[100,200,300,400,500,600,700,800,900,1000],labels=[200,300,400,500,600,700,800,900,1000])
#
# # sparse Matrix of Region code
#
# # # Get one hot encoding of columns B
# # one_hot = pd.get_dummies(data['region_code'])
# # # Drop column B as it is now encoded
# # # df = df.drop('B',axis = 1)
# # # Join the encoded df
# # df = data.join(one_hot)
# # df
#
# #.................................................
# #Group the expenses of creadit card
# data['cc_cons_apr_bin'] = pd.cut(data['cc_cons_apr'],
#              bins=[100,1000,3000,5000,10000,25000,50000,1000000],
#              labels=[1000,3000,5000,10000,25000,50000,1000000])
#
# data['cc_cons_jun_bin'] = pd.cut(data['cc_cons_jun'],
#              bins=[100,1000,3000,5000,10000,25000,50000,1000000],
#              labels=[1000,3000,5000,10000,25000,50000,1000000])
#
# data['cc_cons_may_bin'] = pd.cut(data['cc_cons_may'],
#              bins=[100,1000,3000,5000,10000,25000,50000,1000000],
#              labels=[1000,3000,5000,10000,25000,50000,1000000])
#
#
# test_data['cc_cons_apr_bin'] = pd.cut(test_data['cc_cons_apr'],
#              bins=[100,1000,3000,5000,10000,25000,50000,1000000],
#              labels=[1000,3000,5000,10000,25000,50000,1000000])
#
# test_data['cc_cons_jun_bin'] = pd.cut(test_data['cc_cons_jun'],
#              bins=[100,1000,3000,5000,10000,25000,50000,1000000],
#              labels=[1000,3000,5000,10000,25000,50000,1000000])
#
# test_data['cc_cons_may_bin'] = pd.cut(test_data['cc_cons_may'],
#              bins=[100,1000,3000,5000,10000,25000,50000,1000000],
#              labels=[1000,3000,5000,10000,25000,50000,1000000])
#
# # print(data[['cc_cons_jun','cc_cons_jun_bin']].head())
# # print(data[['cc_cons_apr','cc_cons_apr_bin']].head())
# # print(data[['cc_cons_may','cc_cons_may_bin']].head())
# #
# # print(data[['cc_cons_jun','cc_cons_jun_bin']].sort_values(by = ['cc_cons_jun'],ascending=False).head())
# # print(data[['cc_cons_may','cc_cons_may_bin']].sort_values(by = ['cc_cons_may'],ascending=False).head())
# # print(data[['cc_cons_apr','cc_cons_apr_bin']].sort_values(by = ['cc_cons_apr'],ascending=False).head())
# #
# # print(test_data[['cc_cons_jun','cc_cons_jun_bin']].head())
# # print(test_data[['cc_cons_apr','cc_cons_apr_bin']].head())
# # print(test_data[['cc_cons_may','cc_cons_may_bin']].head())
# #
# # print(test_data[['cc_cons_jun','cc_cons_jun_bin']].sort_values(by = ['cc_cons_jun'],ascending=False).head())
# # print(test_data[['cc_cons_may','cc_cons_may_bin']].sort_values(by = ['cc_cons_may'],ascending=False).head())
# # print(test_data[['cc_cons_apr','cc_cons_apr_bin']].sort_values(by = ['cc_cons_apr'],ascending=False).head())
#
#
# #.................................................
# # Debit card spends in last 3 months
# data['dc_cons_apr'].fillna(0, inplace=True)
# data['dc_cons_apr_bin'] = pd.cut(data['dc_cons_apr'],
#              bins=[-1,0,100,1000,3000,5000,10000,25000,50000,1000000],
#              labels=[0,100,1000,3000,5000,10000,25000,50000,1000000])
#
# data['dc_cons_may'].fillna(0, inplace=True)
# data['dc_cons_may_bin'] = pd.cut(data['dc_cons_may'],
#              bins=[-1,0,100,1000,3000,5000,10000,25000,50000,1000000],
#              labels=[0,100,1000,3000,5000,10000,25000,50000,1000000])
#
# data['dc_cons_jun'].fillna(0, inplace=True)
# data['dc_cons_jun_bin'] = pd.cut(data['dc_cons_jun'],
#              bins=[-1,0,100,1000,3000,5000,10000,25000,50000,1000000],
#              labels=[0,100,1000,3000,5000,10000,25000,50000,1000000])
#
# test_data['dc_cons_apr'].fillna(0, inplace=True)
# test_data['dc_cons_apr_bin'] = pd.cut(test_data['dc_cons_apr'],
#              bins=[-1,0,100,1000,3000,5000,10000,25000,50000,1000000],
#              labels=[0,100,1000,3000,5000,10000,25000,50000,1000000])
#
# test_data['dc_cons_may'].fillna(0, inplace=True)
# test_data['dc_cons_may_bin'] = pd.cut(test_data['dc_cons_may'],
#              bins=[-1,0,100,1000,3000,5000,10000,25000,50000,1000000],
#              labels=[0,100,1000,3000,5000,10000,25000,50000,1000000])
#
# test_data['dc_cons_jun'].fillna(0, inplace=True)
# test_data['dc_cons_jun_bin'] = pd.cut(test_data['dc_cons_jun'],
#              bins=[-1,0,100,1000,3000,5000,10000,25000,50000,1000000],
#              labels=[0,100,1000,3000,5000,10000,25000,50000,1000000])
#
# # print(data[['dc_cons_jun','dc_cons_jun_bin']].head())
# # print(data[['dc_cons_apr','dc_cons_apr_bin']].head())
# # print(data[['dc_cons_may','dc_cons_may_bin']].head())
# #
# # print(data[['dc_cons_jun','dc_cons_jun_bin']].sort_values(by = ['dc_cons_jun'],ascending=False).head())
# # print(data[['dc_cons_may','dc_cons_may_bin']].sort_values(by = ['dc_cons_may'],ascending=False).head())
# # print(data[['dc_cons_apr','dc_cons_apr_bin']].sort_values(by = ['dc_cons_apr'],ascending=False).head())
# #
# # print(test_data[['dc_cons_jun','dc_cons_jun_bin']].head())
# # print(test_data[['dc_cons_apr','dc_cons_apr_bin']].head())
# # print(test_data[['dc_cons_may','dc_cons_may_bin']].head())
# #
# # print(test_data[['dc_cons_jun','dc_cons_jun_bin']].sort_values(by = ['dc_cons_jun'],ascending=False).head())
# # print(test_data[['dc_cons_may','dc_cons_may_bin']].sort_values(by = ['dc_cons_may'],ascending=False).head())
# # print(test_data[['dc_cons_apr','dc_cons_apr_bin']].sort_values(by = ['dc_cons_apr'],ascending=False).head())
# #.................................................
# # count Debit card transactions in last three months
# data['dc_count_apr'].fillna(0, inplace=True)
# data['dc_count_apr_bin'] = pd.cut(data['dc_count_apr'],
#              bins=[-1,0,1,5,20,50,100,5000],
#              labels=[0,1,5,20,50,100,5000])
# print(data[['dc_count_apr','dc_count_apr_bin']].head())
# print(data[['dc_count_apr','dc_count_apr_bin']].sort_values(by = ['dc_count_apr'],ascending=False).head())
#
# data['dc_count_may'].fillna(0, inplace=True)
# data['dc_count_may_bin'] = pd.cut(data['dc_count_may'],
#              bins=[-1,0,1,5,20,50,100,5000],
#              labels=[0,1,5,20,50,100,5000])
# print(data[['dc_count_may','dc_count_may_bin']].head())
# print(data[['dc_count_may','dc_count_may_bin']].sort_values(by = ['dc_count_may'],ascending=False).head())
#
# data['dc_count_jun'].fillna(0, inplace=True)
# data['dc_count_jun_bin'] = pd.cut(data['dc_count_jun'],
#              bins=[-1,0,1,5,20,50,100,5000],
#              labels=[0,1,5,20,50,100,5000])
# print(data[['dc_count_jun','dc_count_jun_bin']].head())
# print(data[['dc_count_jun','dc_count_jun_bin']].sort_values(by = ['dc_count_jun'],ascending=False).head())
#
# # count Credit Card transactions in last three months
# data['cc_count_apr'].fillna(0, inplace=True)
# data['cc_count_apr_bin'] = pd.cut(data['cc_count_apr'],
#              bins=[-1,0,1,5,20,50,100,5000],
#              labels=[0,1,5,20,50,100,5000])
# print(data[['cc_count_apr','cc_count_apr_bin']].head())
# print(data[['cc_count_apr','cc_count_apr_bin']].sort_values(by = ['cc_count_apr'],ascending=False).head())
#
# data['cc_count_may'].fillna(0, inplace=True)
# data['cc_count_may_bin'] = pd.cut(data['cc_count_may'],
#              bins=[-1,0,1,5,20,50,100,5000],
#              labels=[0,1,5,20,50,100,5000])
# print(data[['cc_count_may','cc_count_may_bin']].head())
# print(data[['cc_count_may','cc_count_may_bin']].sort_values(by = ['cc_count_may'],ascending=False).head())
#
# data['cc_count_jun'].fillna(0, inplace=True)
# data['cc_count_jun_bin'] = pd.cut(data['cc_count_jun'],
#              bins=[-1,0,1,5,20,50,100,5000],
#              labels=[0,1,5,20,50,100,5000])
# print(data[['cc_count_jun','cc_count_jun_bin']].head())
# print(data[['cc_count_jun','cc_count_jun_bin']].sort_values(by = ['cc_count_jun'],ascending=False).head())
#
#
#
# # count Debit card transactions in last three months
# test_data['dc_count_apr'].fillna(0, inplace=True)
# test_data['dc_count_apr_bin'] = pd.cut(test_data['dc_count_apr'],
#              bins=[-1,0,1,5,20,50,100,5000],
#              labels=[0,1,5,20,50,100,5000])
# print(test_data[['dc_count_apr','dc_count_apr_bin']].head())
# print(test_data[['dc_count_apr','dc_count_apr_bin']].sort_values(by = ['dc_count_apr'],ascending=False).head())
#
# test_data['dc_count_may'].fillna(0, inplace=True)
# test_data['dc_count_may_bin'] = pd.cut(test_data['dc_count_may'],
#              bins=[-1,0,1,5,20,50,100,5000],
#              labels=[0,1,5,20,50,100,5000])
# print(test_data[['dc_count_may','dc_count_may_bin']].head())
# print(test_data[['dc_count_may','dc_count_may_bin']].sort_values(by = ['dc_count_may'],ascending=False).head())
#
# test_data['dc_count_jun'].fillna(0, inplace=True)
# test_data['dc_count_jun_bin'] = pd.cut(test_data['dc_count_jun'],
#              bins=[-1,0,1,5,20,50,100,5000],
#              labels=[0,1,5,20,50,100,5000])
# print(test_data[['dc_count_jun','dc_count_jun_bin']].head())
# print(test_data[['dc_count_jun','dc_count_jun_bin']].sort_values(by = ['dc_count_jun'],ascending=False).head())
#
# # count Credit Card transactions in last three months
# test_data['cc_count_apr'].fillna(0, inplace=True)
# test_data['cc_count_apr_bin'] = pd.cut(test_data['cc_count_apr'],
#              bins=[-1,0,1,5,20,50,100,5000],
#              labels=[0,1,5,20,50,100,5000])
# print(test_data[['cc_count_apr','cc_count_apr_bin']].head())
# print(test_data[['cc_count_apr','cc_count_apr_bin']].sort_values(by = ['cc_count_apr'],ascending=False).head())
#
# test_data['cc_count_may'].fillna(0, inplace=True)
# test_data['cc_count_may_bin'] = pd.cut(test_data['cc_count_may'],
#              bins=[-1,0,1,5,20,50,100,5000],
#              labels=[0,1,5,20,50,100,5000])
# print(test_data[['cc_count_may','cc_count_may_bin']].head())
# print(test_data[['cc_count_may','cc_count_may_bin']].sort_values(by = ['cc_count_may'],ascending=False).head())
#
# test_data['cc_count_jun'].fillna(0, inplace=True)
# test_data['cc_count_jun_bin'] = pd.cut(test_data['cc_count_jun'],
#              bins=[-1,0,1,5,20,50,100,5000],
#              labels=[0,1,5,20,50,100,5000])
# print(test_data[['cc_count_jun','cc_count_jun_bin']].head())
# print(test_data[['cc_count_jun','cc_count_jun_bin']].sort_values(by = ['cc_count_jun'],ascending=False).head())
#
#
# #.................................................
#
# # Card Limits
# data['card_lim'].fillna(10000,inplace=True)
# data.loc[data['card_lim'] <10000, 'card_lim'] = 10000
# print(data[['id','card_lim']].sort_values(by = ['card_lim'],ascending=False).head())
# print(data[['id','card_lim']].sort_values(by = ['card_lim'],ascending=True).head())
#
# test_data['card_lim'].fillna(10000,inplace=True)
# test_data.loc[test_data['card_lim'] <10000, 'card_lim'] = 10000
# print(test_data[['id','card_lim']].sort_values(by = ['card_lim'],ascending=False).head())
# print(test_data[['id','card_lim']].sort_values(by = ['card_lim'],ascending=True).head())
#
# #.................................................
# # Personal Loan Active
# data['personal_loan_active'].fillna(0,inplace=True)
# data['personal_loan_closed'].fillna(0,inplace=True)
# data['vehicle_loan_active'].fillna(0,inplace=True)
# data['vehicle_loan_closed'].fillna(0,inplace=True)
#
# print(data[['personal_loan_active','personal_loan_closed','vehicle_loan_active','vehicle_loan_closed']].head())
#
# test_data['personal_loan_active'].fillna(0,inplace=True)
# test_data['personal_loan_closed'].fillna(0,inplace=True)
# test_data['vehicle_loan_active'].fillna(0,inplace=True)
# test_data['vehicle_loan_closed'].fillna(0,inplace=True)
# #.................................................
#
# # Investments
#
# data['investment_1'].fillna(0,inplace=True)
# data['investment_2'].fillna(0,inplace=True)
# data['investment_3'].fillna(0,inplace=True)
# data['investment_4'].fillna(0,inplace=True)
#
# test_data['investment_1'].fillna(0,inplace=True)
# test_data['investment_2'].fillna(0,inplace=True)
# test_data['investment_3'].fillna(0,inplace=True)
# test_data['investment_4'].fillna(0,inplace=True)
# print(data['investment_1'])
# # data.loc[data['investment_1'] <=0, 'card_lim'] = 0
# # data.loc[data['investment_2'] <=0, 'card_lim'] = 0
# # data.loc[data['investment_3'] <=0, 'card_lim'] = 0
# # data.loc[data['investment_4'] <=0, 'card_lim'] = 0
# #
# # data.loc[data['investment_1'] <=0, 'card_lim'] = 0
# # data.loc[data['investment_2'] <=0, 'card_lim'] = 0
# # data.loc[data['investment_3'] <=0, 'card_lim'] = 0
# # data.loc[data['investment_4'] <=0, 'card_lim'] = 0

#.............................................
# Loan Enq
data['loan_enq'].fillna(0,inplace=True)
test_data['loan_enq'].fillna(0,inplace=True)
#.............................................

# Credit amount apr

data['tag_col'] = "Train"
test_data["tag_col"] = "Test"
merged_data = pd.concat([data,test_data],sort=False)

# ---------------
bins_x = [-1,0,15000,20000,30000,40000,55000,70000,95000,150000,50000000]
labels_x = [0,15000,20000,30000,40000,55000,70000,95000,150000,50000000]
data = create_bins(data,col='credit_amount_apr',bins=bins_x,labels=labels_x)

bins_x = [-1,0,10000,20000,30000,40000,55000,70000,95000,150000,50000000]
labels_x = [0,10000,20000,30000,40000,55000,70000,95000,150000,50000000]
data = create_bins(data,col='credit_amount_jun',bins=bins_x,labels=labels_x)

bins_x = [-1,0,10000,20000,30000,40000,55000,70000,95000,150000,50000000]
labels_x = [0,10000,20000,30000,40000,55000,70000,95000,150000,50000000]
data = create_bins(data,col='credit_amount_may',bins=bins_x,labels=labels_x)

bins_x = [-1,0,15000,20000,30000,40000,55000,70000,95000,150000,50000000]
labels_x = [0,15000,20000,30000,40000,55000,70000,95000,150000,50000000]
data = create_bins(data,col='credit_amount_apr',bins=bins_x,labels=labels_x)

bins_x = [-1,0,10000,20000,30000,40000,55000,70000,95000,150000,50000000]
labels_x = [0,10000,20000,30000,40000,55000,70000,95000,150000,50000000]
data = create_bins(data,col='credit_amount_jun',bins=bins_x,labels=labels_x)

bins_x = [-1,0,10000,20000,30000,40000,55000,70000,95000,150000,50000000]
labels_x = [0,10000,20000,30000,40000,55000,70000,95000,150000,50000000]
data = create_bins(data,col='credit_amount_may',bins=bins_x,labels=labels_x)

# Max credt amount
bins_x = [-1,0,15000,20000,30000,40000,55000,70000,95000,150000,50000000]
labels_x = [0,15000,20000,30000,40000,55000,70000,95000,150000,50000000]
data = create_bins(data,col='max_credit_amount_apr',bins=bins_x,labels=labels_x)

bins_x = [-1,0,10000,20000,30000,40000,55000,70000,95000,150000,50000000]
labels_x = [0,10000,20000,30000,40000,55000,70000,95000,150000,50000000]
data = create_bins(data,col='max_credit_amount_jun',bins=bins_x,labels=labels_x)

bins_x = [-1,0,10000,20000,30000,40000,55000,70000,95000,150000,50000000]
labels_x = [0,10000,20000,30000,40000,55000,70000,95000,150000,50000000]
data = create_bins(data,col='max_credit_amount_may',bins=bins_x,labels=labels_x)

#/........................
# Credit and Debit Count
bins_x = [-1,0,1,2,4,6,20,500]
labels_x = [0,1,2,4,6,20,500]
data = create_bins(data,col='credit_count_may',bins=bins_x,labels=labels_x)
data = create_bins(data,col='credit_count_jun',bins=bins_x,labels=labels_x)
data = create_bins(data,col='credit_count_apr',bins=bins_x,labels=labels_x)
data = create_bins(data,col='debit_count_may',bins=bins_x,labels=labels_x)
data = create_bins(data,col='debit_count_apr',bins=bins_x,labels=labels_x)
data = create_bins(data,col='debit_count_jun',bins=bins_x,labels=labels_x)


