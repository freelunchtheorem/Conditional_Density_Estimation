import os
from datetime import datetime

def store_dataframe(dataframe, output_dir, file_name=None):
  assert os.path.exists(output_dir) or os.path.exists(os.path.abspath(os.path.join(os.getcwd(), output_dir))), "invalid path to output directory"
  if file_name is None:
    file_name = "evaluated_configs_df_"+ str(datetime.now().strftime("%m-%d-%y_%H-%M-%S")) +'.pickle'
    full_path = os.path.join(output_dir, file_name)
  else:
    full_path = os.path.join(output_dir, file_name)
  dataframe.to_pickle(full_path)
  print("Dumped dataframe pickle to ", full_path)


def store_csv(dataframe, output_dir, file_name=None):
  assert os.path.exists(output_dir) or os.path.exists(os.path.abspath(os.path.join(os.getcwd(), output_dir))), "invalid path to output directory"
  if file_name is None:
    file_name = "evaluated_configs_df_" + str(datetime.now().strftime("%m-%d-%y_%H-%M-%S")) + '.csv'
    full_path = os.path.join(output_dir, file_name)
  else:
    full_path = os.path.join(output_dir, file_name)
  dataframe.to_csv("./a.csv", sep=';', header=True)
  print("Dumped csv to ", full_path)

