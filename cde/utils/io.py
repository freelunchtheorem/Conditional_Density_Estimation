import os
import pickle
import pandas as pd
from datetime import datetime


def store_dataframe(dataframe, output_dir, file_name=None):
  suffix = ".pickle"
  full_path = get_full_path(output_dir=output_dir, suffix=suffix, file_name=file_name)
  dataframe.to_pickle(full_path)
  print("Dumped dataframe pickle to", full_path)


def store_csv(dataframe, output_dir, file_name=None):
  suffix = ".csv"
  full_path = get_full_path(output_dir=output_dir, suffix=suffix, file_name=file_name)

  dataframe.to_csv(full_path, sep=';', header=True)
  print("Dumped dataframe csv to", full_path)


def store_objects(objs, output_dir, file_name=None):
  suffix = ".pickle"
  full_path = get_full_path(output_dir=output_dir, suffix=suffix, file_name=file_name)

  with open(full_path, 'wb') as output_file:
    pickle.dump(objs, output_file)
  print("Dumped pickle to", full_path)


def dump_as_pickle(file_handle, obj, verbose=False):
  try:
    pickle.dump(obj=obj, file=file_handle)
    if verbose:
      print("successfully dumped pickle file to {}".format(os.path.abspath(file_handle.name)))
  except Exception as e:
    print("dumping pickle object not successful")
    print(str(e))
  return True


def append_result_to_csv(file_handle, result):
  if file_handle.closed:
    return False
  try:
    if os.stat(file_handle.name).st_size == 0: # checks if csv file is empty
      result.to_csv(file_handle.name, sep=';', header=True, mode='a', index=False)
    else:
      result.to_csv(file_handle.name, sep=';', header=False, mode='a', index=False)
  except Exception as e:
    print("appending to csv not successful")
    print(str(e))
  return True

def get_full_path(output_dir, suffix=".pickle", file_name=None):
  assert os.path.exists(output_dir) or os.path.exists(os.path.abspath(os.path.join(os.getcwd(), output_dir))), "invalid path to output directory"
  if file_name is None:
    file_name = str(datetime.now().strftime("%m-%d-%y_%H-%M-%S")) + str(suffix)
    full_path = os.path.join(output_dir, file_name)
  else:
    full_path = os.path.join(output_dir, file_name + str(datetime.now().strftime("%m-%d-%y_%H-%M-%S")) + str(suffix))
  return full_path


def load_time_series_csv(file_path, delimiter=',', time_format=None, time_columns=None):
  """ Loads a .csv time series file (e.g. EuroStoxx50) as a pandas dataframe and applies some basic formatting.
  The basic formatting includes:
  a) if no time column is available in the .csv, calling this function sorts the data according to the first column
  b) if a time column is available (i.e. some column containing the string 'time'), the function tries to re-arrange the column into an
  expected format, sets it as an index and sorts it according to the date stamps


  Args:
    file_path: an absolute or relative path to the .csv file as str
    delimiter: the column separator used in the .csv file
    time_format: optional but if set (must be str), the function tries to re-arrange the date column into a deviating format
    time_columns: optional list of strings indicating the names of the time columns within the csv file

  Returns:
    a pandas dataframe containing the information from the .csv file. If a time or date colum is available, the df contains the date as
    index and is sorted according to this column.
  """
  assert os.path.exists(file_path), "invalid path to output directory"
  time_series = pd.read_csv(file_path, delimiter=delimiter)

  if time_format is None:
    POSSIBLE_TIME_FORMATS = ['%Y-%m-%d %H:%M:%S', '%d-%m-%y %H:%M:%S', "%Y%m%d"]
  else:
    POSSIBLE_TIME_FORMATS = [time_format]

  columns = list(time_series.columns.values)
  if time_columns is None:
    TIME_COLUMNS = ['time', 'date']
  else:
    TIME_COLUMNS = time_columns
  #time_col = [s for s in columns if "time" in s]
  time_col = [s for s in columns for t in TIME_COLUMNS if t in s]


  if time_col:
    time_col = time_col[0] # take first occurrence
    for format in POSSIBLE_TIME_FORMATS:
      try:
        time_series[time_col] = pd.to_datetime(time_series[time_col], format=format)  # try to get the date
        break  # if correct format, don't test any other formats
      except ValueError:
        pass  # if incorrect format, keep trying other formats

    time_series = time_series.set_index(time_col)

  time_series = time_series.sort_index()
  return time_series



