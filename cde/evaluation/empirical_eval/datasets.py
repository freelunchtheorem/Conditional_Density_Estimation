from urllib.request import urlretrieve
import os
import pandas as pd
import numpy as np
from datetime import datetime

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data'))

UCI_DATASETS = []

def _UCI(C):
    UCI_DATASETS.append(C)
    return C

class Dataset:

    ndim_x = None
    ndim_y = None

    target_columns = []
    feature_columns = []

    data_file_name = ''
    download_url = ''

    @property
    def data_file_path(self):
        return os.path.join(DATA_DIR, self.data_file_name)

    @property
    def needs_download(self):
        return not os.path.isfile(self.data_file_path)

    def download_dataset(self):
        print("Downloading data file from %s" % self.download_url)
        urlretrieve(self.download_url, self.data_file_path)

    def get_df(self):
        if self.needs_download:
            self.download_dataset()
        df = pd.read_csv(self.data_file_path)
        return self._process_df(df)

    def get_target_feature_split(self):
        df = self.get_df()
        X = np.array(df[self.feature_columns])
        Y = np.array(df[self.target_columns])
        assert X.ndim == Y.ndim == 2
        assert X.shape[0] == Y.shape[0]
        assert X.shape[1] == self.ndim_x and Y.shape[1] == self.ndim_y
        return X, Y

    def get_train_valid_splits(self, valid_portion, n_splits, shuffle=False, random_state=None):

        X, Y = self.get_target_feature_split()

        n_instances = X.shape[0]
        valid_size = int(valid_portion * n_instances)
        assert valid_size * n_splits <= n_instances

        idx = np.arange(n_instances)
        if shuffle:
            if random_state is not None:
                random_state.shuffle(idx)
            else:
                np.random.shuffle(idx)

        X_trains, Y_trains, X_valids, Y_valids = [], [], [], []

        for i in reversed(range(n_splits)):
            idx_start = (n_instances // n_splits) * i
            idx_end = idx_start + valid_size

            idx_train, idx_valid = np.concatenate([idx[:idx_start], idx[idx_end:]]),  idx[idx_start:idx_end]

            assert len(set(idx_train) | set(idx_valid)) == n_instances

            X_trains.append(X[idx_train, :])
            Y_trains.append(Y[idx_train, :])
            X_valids.append(X[idx_valid, :])
            Y_valids.append(Y[idx_valid, :])

        return X_trains, Y_trains, X_valids, Y_valids

    def _process_df(self, df):
        return df

    def __str__(self):
        return "%s (ndim_x = %i, ndim_y = %i)"%(str(self.__class__.__name__), self.ndim_x, self.ndim_y)


class EuroStoxx50(Dataset):

    ndim_x = 14
    ndim_y = 1

    target_columns = ['log_ret_1']
    feature_columns = ['log_ret_last_period', 'log_risk_free_1d',
       'RealizedVariation', 'bakshiSkew', 'bakshiKurt', 'SVIX', 'Mkt-RF',
       'SMB', 'HML', 'WML', 'WML_risk_10d', 'Mkt-RF_risk_10d', 'SMB_risk_10d',
       'HML_risk_10d']

    data_file_name = 'eurostoxx50.csv'

    def get_train_valid_splits(self, valid_portion, n_splits, shift_size=100, shuffle=False, random_state=None):
        # needs extra treatment since it's time-series data --> shifts train and valid set by shift_size each split
        # --> ensures that the valid data is always in the future of the train data

        assert shuffle is False

        X, Y = self.get_target_feature_split()

        n_instances = X.shape[0]
        valid_size = int(valid_portion * n_instances)
        training_size = n_instances - valid_size - n_splits*shift_size
        assert valid_size * n_splits <= n_instances

        idx = np.arange(n_instances)

        X_trains, Y_trains, X_valids, Y_valids = [], [], [], []

        for i in reversed(range(n_splits)):
            idx_train_start = int(i * shift_size)
            idx_valid_start = idx_train_start + training_size
            idx_valid_end = idx_valid_start + valid_size

            idx_train, idx_valid = idx[idx_train_start:idx_valid_start], idx[idx_valid_start:idx_valid_end]


            X_trains.append(X[idx_train, :])
            Y_trains.append(Y[idx_train, :])
            X_valids.append(X[idx_valid, :])
            Y_valids.append(Y[idx_valid, :])

        return X_trains, Y_trains, X_valids, Y_valids

    def download_dataset(self):
        raise AssertionError("Sry, the EuroStoxx 50 data is proprietary and won't be open-sourced")

class NCYTaxiDropoffPredict(Dataset):

    ndim_x = 6
    ndim_y = 2

    data_file_name = 'yellow_tripdata_2016-01.csv'
    data_file_name_processed = 'yellow_tipdata_2016-01_processed.csv'
    download_url = 'https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2016-01.csv'

    target_columns = ['dropoff_loc_lat', 'dropoff_loc_lon']
    feature_columns = ['pickup_loc_lat', 'pickup_loc_lon', 'pickup_time_day_of_week_sin', 'pickup_time_day_of_week_cos',
                       'pickup_time_of_day_sin', 'pickup_time_of_day_cos']

    x_bounds = [-74.04, -73.75]
    y_bounds = [40.62, 40.86]
    too_close_radius = 0.00001
    min_duration = 30
    max_duration = 3 * 3600

    def __init__(self, n_samples=10**4, seed=22):
        self.n_samples = n_samples
        self.random_state = np.random.RandomState(seed)

    def get_df(self):
        data_file_path_processed = os.path.join(DATA_DIR, self.data_file_name_processed)
        if not os.path.isfile(data_file_path_processed):
            df = super(NCYTaxiDropoffPredict, self).get_df().dropna()
            print("save processed NYC data as csv to %s" % data_file_path_processed)
            df.to_csv(data_file_path_processed)            

        print("loading %s" % data_file_path_processed)
        df = pd.read_csv(data_file_path_processed)

        return df.sample(n=self.n_samples, random_state=self.random_state)

    def _process_df(self, df): # does some data cleaning
        data = df.values

        pickup_loc = np.array((data[:, 5], data[:, 6])).T
        dropoff_loc = np.array((data[:, 9], data[:, 10])).T

        ind = np.ones(len(data)).astype(bool)
        ind[data[:, 5] < self.x_bounds[0]] = False
        ind[data[:, 5] > self.x_bounds[1]] = False
        ind[data[:, 6] < self.y_bounds[0]] = False
        ind[data[:, 6] > self.y_bounds[1]] = False

        ind[data[:, 9] < self.x_bounds[0]] = False
        ind[data[:, 9] > self.x_bounds[1]] = False
        ind[data[:, 10] < self.y_bounds[0]] = False
        ind[data[:, 10] > self.y_bounds[1]] = False

        print('discarding {} out of bounds {} {}'.format(np.sum(np.invert(ind).astype(int)), self.x_bounds, self.y_bounds))

        early_stop = ((data[:, 5] - data[:, 9]) ** 2 + (data[:, 6] - data[:, 10]) ** 2 < self.too_close_radius)
        ind[early_stop] = False
        print('discarding {} trip less than {} gp dist'.format(np.sum(early_stop.astype(int)), self.too_close_radius ** 0.5))

        times = np.array([_process_time(d_pickup, d_dropoff) for (d_pickup, d_dropoff) in data[:, 1:3]])
        pickup_time = times[:, :2]
        dropoff_time = times[:, 2:4]
        duration = times[:, 4]

        short_journeys = (duration < self.min_duration)
        ind[short_journeys] = False
        print('discarding {} less than {}s journeys'.format(np.sum(short_journeys.astype(int)), self.min_duration))

        long_journeys = (duration > self.max_duration)
        ind[long_journeys] = False
        print('discarding {} more than {}h journeys'.format(np.sum(long_journeys.astype(int)), self.max_duration / 3600.))

        pickup_loc_lat = pickup_loc[ind, 0]
        pickup_loc_lon = pickup_loc[ind, 1]

        dropoff_loc_lat = dropoff_loc[ind, 0]
        dropoff_loc_lon = dropoff_loc[ind, 1]

        pickup_time_day_of_week = pickup_time[ind, 0]
        pickup_time_of_day = pickup_time[ind, 1]

        dropoff_time_day_of_week = dropoff_time[ind, 0]
        dropoff_time_of_day = dropoff_time[ind, 1]

        duration = duration[ind]

        print('{} total rejected journeys'.format(np.sum(np.invert(ind).astype(int))))

        df_processed = pd.DataFrame(
            {"pickup_loc_lat": pickup_loc_lat,
             "pickup_loc_lon": pickup_loc_lon,
             "dropoff_loc_lat": dropoff_loc_lat,
             "dropoff_loc_lon": dropoff_loc_lon,
             "pickup_time_day_of_week": pickup_time_day_of_week.astype(np.int),
             "pickup_time_day_of_week_sin": np.sin(pickup_time_day_of_week),
             "pickup_time_day_of_week_cos": np.cos(pickup_time_day_of_week.astype(np.int)),
             "pickup_time_of_day": pickup_time_of_day,
             "pickup_time_of_day_sin": np.sin(pickup_time_of_day),
             "pickup_time_of_day_cos": np.cos(pickup_time_of_day),
             "dropoff_time_day_of_week": dropoff_time_day_of_week.astype(np.int),
             "dropoff_time_of_day": dropoff_time_of_day, "duration": duration})

        return df_processed

    def __str__(self):
        return "%s (n_samples = %i, ndim_x = %i, ndim_y = %i)" % (str(self.__class__.__name__), self.n_samples, self.ndim_x, self.ndim_y)

class UCI_Dataset(Dataset):
    uci_base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'
    uci_data_path = ''

    @property
    def download_url(self):
        return os.path.join(self.uci_base_url, self.uci_data_path)

    @property
    def target_columns(self):
        return [self.get_df().columns[-1]]

    @property
    def feature_columns(self):
        return list(self.get_df().columns[:-1])

@_UCI
class BostonHousing(UCI_Dataset):
    uci_data_path = 'housing/housing.data'
    data_file_name = 'housing.data'

    ndim_x = 13
    ndim_y = 1

    def get_df(self):
        if self.needs_download:
            self.download_dataset()
        df = pd.read_fwf(self.data_file_path, header=None)
        return df

@_UCI
class Conrete(UCI_Dataset):
    uci_data_path = 'concrete/compressive/Concrete_Data.xls'
    data_file_name = 'concrete.xls'

    ndim_y = 1
    ndim_x = 8

    def get_df(self):
        if self.needs_download:
            self.download_dataset()
        df = pd.read_excel(self.data_file_path).dropna()
        return df

@_UCI
class Energy(UCI_Dataset):
    uci_data_path ='00242/ENB2012_data.xlsx'
    data_file_name = 'energy.xlsx'

    ndim_x = 9
    ndim_y = 1

    def get_df(self):
        if self.needs_download:
            self.download_dataset()
        df = pd.read_excel(self.data_file_path).dropna()
        return df

@_UCI
class Power(UCI_Dataset):
    download_url = 'https://www.dropbox.com/s/w7qkzjtuynwxjke/power.csv?dl=1'
    data_file_name = 'power.csv'

    ndim_x = 4
    ndim_y = 1

    def get_df(self):
        if self.needs_download:
            self.download_dataset()
        df = pd.read_csv(self.data_file_path).dropna()
        return df

@_UCI
class Protein(UCI_Dataset):
    uci_data_path = '00265/CASP.csv'
    data_file_name = 'protein.csv'

    ndim_x = 9
    ndim_y = 1

    def get_df(self):
        if self.needs_download:
            self.download_dataset()
        df = pd.read_csv(self.data_file_path).dropna()
        return df

@_UCI
class WineRed(UCI_Dataset):
    uci_data_path = 'wine-quality/winequality-red.csv'
    data_file_name = 'wine_red.csv'

    ndim_x = 11
    ndim_y = 1

    def get_df(self):
        if self.needs_download:
            self.download_dataset()
        df = pd.read_csv(self.data_file_path, delimiter=';').dropna()
        return df

@_UCI
class WineWhite(UCI_Dataset):
    uci_data_path = 'wine-quality/winequality-white.csv'
    data_file_name = 'wine_white.csv'

    ndim_x = 11
    ndim_y = 1

    def get_df(self):
        if self.needs_download:
            self.download_dataset()
        df = pd.read_csv(self.data_file_path, delimiter=';').dropna()
        return df

@_UCI
class Yacht(UCI_Dataset):
    uci_data_path = '00243/yacht_hydrodynamics.data'
    data_file_name = 'yacht.data'

    ndim_x = 6
    ndim_y = 1

    def get_df(self):
        if self.needs_download:
            self.download_dataset()
        df = pd.read_fwf(self.data_file_path, header=None).dropna()
        return df


""" helper methods """

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


if __name__ == "__main__":
    for dataset_class in [EuroStoxx50, NCYTaxiDropoffPredict] + UCI_DATASETS:
        dataset = dataset_class()
        _, Y = dataset.get_target_feature_split()
        n_samples = Y.shape[0]
        print("%s: n_samples = %i, ndim_x = %i, ndim_y = %i"%(str(dataset.__class__.__name__), n_samples, dataset.ndim_x, dataset.ndim_y))