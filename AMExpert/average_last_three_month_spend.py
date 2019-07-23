import pandas as pd
from sklearn.metrics import mean_squared_log_error

# Read the data
data = pd.read_csv("../Train/train.csv")
test_data = pd.read_csv("../test.csv")

print(data.columns)

train_predictions = data.cc_cons_apr+data.cc_cons_may+data.cc_cons_jun
print(mean_squared_log_error(data.cc_cons,train_predictions))

test_predictions = test_data.cc_cons_apr+test_data.cc_cons_may+test_data.cc_cons_jun
final_submission_data = pd.DataFrame({'id':test_data.id,'cc_cons':test_predictions})
print(final_submission_data.head())
final_submission_data.to_csv("../Submissions/avg_submission.csv",index=False)