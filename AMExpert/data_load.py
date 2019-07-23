import pandas as pd
from pathlib import Path
import numpy as np

class read:
    def __init__(self):
        pass

    def read_files(self, dir_path,type):
        data_dir = Path(dir_path)
        full_df = pd.concat(
            pd.read_parquet(parquet_file)
            for parquet_file in data_dir.glob('*.'+type)
            )
        return full_df

    def data_summary(self,data:pd.DataFrame):
        result = {"Shape":data.shape,"Datatype":data.dtypes}
        print(result)
        return result

    def na_check(self,pandas_data:pd.DataFrame):
        data_with_null = pandas_data.isnull()
        if data_with_null.values.any():
            result_data = data_with_null.sum()
        else:
            result_data = pd.DataFrame()
            print("No Missing Data")
        return result_data

    def return_column_on_type(self, data:pd.DataFrame,type):
        cols = data.columns[data.dtypes.eq(type)]
        return(cols)

    def describe_data(self,data:pd.DataFrame):
        # print(data.dtypes.iloc[:1])
        print(data.dtypes)
        if np.number in data.dtypes:
            print("Numeric Values Present")

        # res_numeric = data.describe(include=np.number)
        # res_object = data.describe(include = 'object')
        # res_cat = data.describe(include='category')
        # return(res_numeric,res_object,res_cat)


