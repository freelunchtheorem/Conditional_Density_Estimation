import os
import pickle
from datetime import datetime


def store_dataframe(dataframe, output_dir, file_name=None):
  assert os.path.exists(output_dir) or os.path.exists(os.path.abspath(os.path.join(os.getcwd(), output_dir))), "invalid path to output directory"
  if file_name is None:
    file_name = str(datetime.now().strftime("%m-%d-%y_%H-%M-%S")) +'.pickle'
    full_path = os.path.join(output_dir, file_name)
  else:
    full_path = os.path.join(output_dir, file_name + str(datetime.now().strftime("%m-%d-%y_%H-%M-%S")) +'.pickle')

  dataframe.to_pickle(full_path)
  print("Dumped dataframe pickle to", full_path)


def store_csv(dataframe, output_dir, file_name=None):
  assert os.path.exists(output_dir) or os.path.exists(os.path.abspath(os.path.join(os.getcwd(), output_dir))), "invalid path to output directory"
  if file_name is None:
    file_name = str(datetime.now().strftime("%m-%d-%y_%H-%M-%S")) + '.csv'
    full_path = os.path.join(output_dir, file_name)
  else:
    full_path = os.path.join(output_dir, file_name+ str(datetime.now().strftime("%m-%d-%y_%H-%M-%S")) + '.csv')

  dataframe.to_csv(full_path, sep=';', header=True)
  print("Dumped dataframe csv to", full_path)


def store_objects(objs, output_dir, file_name=None):
  assert os.path.exists(output_dir) or os.path.exists(os.path.abspath(os.path.join(os.getcwd(), output_dir))), "invalid path to output directory"
  if file_name is None:
    file_name = str(datetime.now().strftime("%m-%d-%y_%H-%M-%S")) + '.pickle'
    full_path = os.path.join(output_dir, file_name)
  else:
    full_path = os.path.join(output_dir, file_name + str(datetime.now().strftime("%m-%d-%y_%H-%M-%S")) + '.pickle')

  with open(full_path, 'wb') as output_file:
    pickle.dump(objs, output_file)
  print("Dumped pickle to", full_path)
