import numpy as np
import os
import pandas as pd
from datetime import datetime

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data'))

NYC_YELLOW_TAXI_2016 = [
                  os.path.join(DATA_DIR, "yellow_tripdata_2016-01.csv"),
                  #os.path.join(DATA_DIR, "yellow_tripdata_2018-02.csv")
                ]


def _convert_to_day_minute(d):
  rescale = lambda x, a, b: b[0] + (b[1] - b[0]) * x / (a[1] - a[0])

  day_of_week = rescale(float(d.weekday()), [0, 6], [0, 2 * np.pi])
  time_of_day = rescale(d.time().hour * 60 + d.time().minute, [0, 24 * 60], [0, 2 * np.pi])
  return day_of_week, time_of_day


def _process_time(pickup_datetime, dropoff_datetime):
  d_pickup = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")
  d_dropoff = datetime.strptime(dropoff_datetime, "%Y-%m-%d %H:%M:%S")
  duration = (d_dropoff - d_pickup).total_seconds()

  pickup_day_of_week, pickup_time_of_day = _convert_to_day_minute(d_pickup)
  dropoff_day_of_week, dropoff_time_of_day = _convert_to_day_minute(d_dropoff)

  return [pickup_day_of_week, pickup_time_of_day, dropoff_day_of_week, dropoff_time_of_day, duration]


""" PUBLIC METHODS """
def make_overall_nyc_yellow_taxi_df():
  x_bounds = [-74.04, -73.75]
  y_bounds = [40.62, 40.86]
  too_close_radius = 0.00001
  min_duration = 30
  max_duration = 3 * 3600

  df = pd.DataFrame([])

  for taxi_csv_month in NYC_YELLOW_TAXI_2016:
    data = pd.read_csv(taxi_csv_month)
    data = data.values

    # print(data.dtypes.index)
    # 'vendor_id',  0
    # 'pickup_datetime', 1
    # 'dropoff_datetime',2
    # 'passenger_count', 3
    # 'trip_distance', 4'
    # 'pickup_longitude', 5
    # 'pickup_latitude', 6
    # 'RatecodeID', 7
    # 'store_and_fwd_flag', 8,
    # 'dropoff_longitude', 9
    # 'dropoff_latitude', 10
    # ...

    pickup_loc = np.array((data[:, 5], data[:, 6])).T
    dropoff_loc = np.array((data[:, 9], data[:, 10])).T

    ind = np.ones(len(data)).astype(bool)
    ind[data[:, 5] < x_bounds[0]] = False
    ind[data[:, 5] > x_bounds[1]] = False
    ind[data[:, 6] < y_bounds[0]] = False
    ind[data[:, 6] > y_bounds[1]] = False

    ind[data[:, 9] < x_bounds[0]] = False
    ind[data[:, 9] > x_bounds[1]] = False
    ind[data[:, 10] < y_bounds[0]] = False
    ind[data[:, 10] > y_bounds[1]] = False

    print('discarding {} out of bounds {} {}'.format(np.sum(np.invert(ind).astype(int)), x_bounds, y_bounds))

    early_stop = ((data[:, 5] - data[:, 9]) ** 2 + (data[:, 6] - data[:, 10]) ** 2 < too_close_radius)
    ind[early_stop] = False
    print('discarding {} trip less than {} gp dist'.format(np.sum(early_stop.astype(int)), too_close_radius ** 0.5))

    times = np.array([_process_time(d_pickup, d_dropoff) for (d_pickup, d_dropoff) in data[:, 1:3]])
    pickup_time = times[:, :2]
    dropoff_time = times[:, 2:4]
    duration = times[:, 4]

    short_journeys = (duration < min_duration)
    ind[short_journeys] = False
    print('discarding {} less than {}s journeys'.format(np.sum(short_journeys.astype(int)), min_duration))

    long_journeys = (duration > max_duration)
    ind[long_journeys] = False
    print('discarding {} more than {}h journeys'.format(np.sum(long_journeys.astype(int)), max_duration / 3600.))

    pickup_loc_lat = pickup_loc[ind, 0]
    pickup_loc_lon = pickup_loc[ind, 1]

    dropoff_loc_lat = dropoff_loc[ind, 0]
    dropoff_loc_lon = dropoff_loc[ind, 1]

    pickup_time_day_of_week = pickup_time[ind, 0]  # first column: pickup day of week (4--> Friday), pickup time of day
    pickup_time_of_day = pickup_time[ind, 1]

    dropoff_time_day_of_week = dropoff_time[ind, 0]
    dropoff_time_of_day = dropoff_time[ind, 1]

    duration = duration[ind]

    print('{} total rejected journeys'.format(np.sum(np.invert(ind).astype(int))))

    df = df.append(pd.DataFrame(
      {"pickup_loc_lat": pickup_loc_lat, "pickup_loc_lon": pickup_loc_lon, "dropoff_loc_lat": dropoff_loc_lat, "dropoff_loc_lon": dropoff_loc_lon,
       "pickup_time_day_of_week": pickup_time_day_of_week.astype(np.int), "pickup_time_of_day": pickup_time_of_day, "dropoff_time_day_of_week": dropoff_time_day_of_week.astype(np.int),
       "dropoff_time_of_day": dropoff_time_of_day, "duration": duration}), ignore_index=True)

  return df


if __name__ == '__main__':
  df = make_overall_nyc_yellow_taxi_df()
  #X, Y = target_feature_split(df, 'log_ret_1', filter_nan=True)
  #print(X, Y)