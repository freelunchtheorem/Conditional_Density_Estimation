import os
import pickle
from datetime import datetime
from pathlib import Path


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


def append_obj_to_pickle(file_handle, obj):
  if file_handle.closed:
    return False
  pickle.dump(obj=obj, file=file_handle)
  return True


def append_result_to_csv(file_handle, result, index):
  if file_handle.closed:
    return False
  if index == 0:
    result.to_csv(file_handle, sep=';', header=True)
  else:
    result.to_csv(file_handle, sep=';', header=False, mode="a")
  return True


def get_full_path(output_dir, suffix=".pickle", file_name=None):
  assert os.path.exists(output_dir) or os.path.exists(os.path.abspath(os.path.join(os.getcwd(), output_dir))), "invalid path to output directory"
  if file_name is None:
    file_name = str(datetime.now().strftime("%m-%d-%y_%H-%M-%S")) + str(suffix)
    full_path = os.path.join(output_dir, file_name)
  else:
    full_path = os.path.join(output_dir, file_name + str(datetime.now().strftime("%m-%d-%y_%H-%M-%S")) + str(suffix))
  return full_path