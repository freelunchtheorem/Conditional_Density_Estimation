import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cde.utils.io import load_time_series_csv

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))

class TestIO(unittest.TestCase):

  def test_loading_time_series(self):
    to_test = [os.path.join(DATA_DIR, "2_Eurostoxx50/eur_ois.csv"),
               os.path.join(DATA_DIR, "2_Eurostoxx50/eurostoxx50_prices_eod.csv"),
               os.path.join(DATA_DIR, "2_Eurostoxx50/eurostoxx50_exp_tail_variation_measures.csv"),
               os.path.join(DATA_DIR, "2_Eurostoxx50/eurostoxx50_realized_volmeasures.csv"),
              [os.path.join(DATA_DIR,"2_Eurostoxx50/eurostoxx50_riskneutralmeasures.csv"), ";"],
              [os.path.join(DATA_DIR, "2_Eurostoxx50/eurostoxx50_vrp.csv"), ";"],
              [os.path.join(DATA_DIR, "2_Eurostoxx50/FamaFrench_Europe_3_Factors_Daily.csv"), ",", "%Y%m%d"],
              [os.path.join(DATA_DIR, "2_Eurostoxx50/FamaFrench_Europe_MOM_Factor_Daily.csv"), ",", "%Y%m%d"],
             ]


    for charge in to_test:
      if type(charge) == list:
        self.assertFalse(load_time_series_csv(*charge).empty)
      else:
        self.assertFalse(load_time_series_csv(charge).empty)




if __name__ == '__main__':

  testmodules = ['unittests_io.TestIO']
  suite = unittest.TestSuite()
  for t in testmodules:
    try:
      # If the module defines a suite() function, call it to get the suite.
      mod = __import__(t, globals(), locals(), ['suite'])
      suitefn = getattr(mod, 'suite')
      suite.addTest(suitefn())
    except (ImportError, AttributeError):
      # else, just load all the test cases from the module.
      suite.addTest(unittest.defaultTestLoader.loadTestsFromName(t))

  unittest.TextTestRunner().run(suite)