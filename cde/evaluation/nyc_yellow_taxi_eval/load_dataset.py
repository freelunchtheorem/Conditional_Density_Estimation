import numpy as np
import os
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline

from cde.utils.io import load_time_series_csv


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data'))

NYC_YELLOW_TAXI_2018 = [
                  os.path.join(DATA_DIR, "yellow_tripdata_2018-01.csv"),
                  os.path.join(DATA_DIR, "yellow_tripdata_2018-02.csv")
                ]


""" PUBLIC METHODS """
def make_overall_nyc_yellow_taxi_df():
  time_columns = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
  dfs = []
  for month in NYC_YELLOW_TAXI_2018:
    df_month = load_time_series_csv(month, time_columns=time_columns)
    df_month = df_month.loc["2018-01-01":]  # dataset contains some samples with data previous of the specified dataset date
    dfs.append(df_month)

  df = pd.concat(dfs, axis=0)

  return df



if __name__ == '__main__':
  df = make_overall_nyc_yellow_taxi_df()
  #X, Y = target_feature_split(df, 'log_ret_1', filter_nan=True)
  #print(X, Y)