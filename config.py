import os

''' Directories '''
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'local')


''' Personal configuration '''

# If there is a a config_personal.py file , import this file
config_personal_path = os.path.join(BASE_DIR, 'config_personal.py')
if os.path.isfile(config_personal_path):
    from config_personal import *