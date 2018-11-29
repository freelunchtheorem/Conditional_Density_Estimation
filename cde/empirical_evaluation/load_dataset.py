from cde.utils.io import load_time_series_csv
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

EURO_OIS_CSV = "../../data/2_Eurostoxx50/eur_ois.csv"
EUROSTOXX_CSV = "../../data/2_Eurostoxx50/eurostoxx50_prices_eod.csv"
EURO_TAIL_VARIATION_CSV = "../../data/2_Eurostoxx50/eurostoxx50_exp_tail_variation_measures.csv"
REALIZED_VOL_CSV = "../../data/2_Eurostoxx50/eurostoxx50_realized_volmeasures.csv"
RISKNEUTRAL_CSV = "../../data/2_Eurostoxx50/eurostoxx50_riskneutralmeasures.csv"
VRP_CSV = "../../data/2_Eurostoxx50/eurostoxx50_vrp.csv"
FAMA_FRENCH_CSV = "../../data/2_Eurostoxx50/FamaFrench_Europe_3_Factors_Daily.csv"
FAMA_FRENCH_MOMENTS_CSV = "../../data/2_Eurostoxx50/FamaFrench_Europe_MOM_Factor_Daily.csv"

def make_return_df(return_periods):
  eurostoxx = load_time_series_csv(EUROSTOXX_CSV)
  for h in return_periods:
    eurostoxx['log_ret_%i'%h] = np.log(eurostoxx.lastprice) - np.log(eurostoxx.lastprice.shift(h))
  return eurostoxx.drop(labels=['lastprice'], axis=1)


def make_risk_free_df():
  euro_oid = load_time_series_csv(EURO_OIS_CSV)
  euro_oid = euro_oid[euro_oid.maturity == 1]
  euro_oid['log_risk_free_1d'] = np.log((euro_oid['yield']/365) + 1)
  return euro_oid.drop(labels=['maturity', 'yield'], axis=1)

def make_exp_tail_variation_df():
  exp_tail_variation = load_time_series_csv(EURO_TAIL_VARIATION_CSV)
  return exp_tail_variation

def make_realized_vol_df():
  realized_vol = load_time_series_csv(REALIZED_VOL_CSV)
  return realized_vol.loc[:, ['RealizedVariation']]

def make_riskneutral_df():
  riskteural_measures = load_time_series_csv(RISKNEUTRAL_CSV, delimiter=';')
  return riskteural_measures.loc[:, ['bakshiSkew', 'bakshiKurt', 'SVIX']]

def make_variance_risk_premium_df():
  vrp = load_time_series_csv(VRP_CSV, delimiter=';')
  return vrp

def make_fama_french_df():
  fama_french_factors = load_time_series_csv(FAMA_FRENCH_CSV, time_format="%Y%m%d")
  return fama_french_factors.loc[:, ['Mkt-RF', 'SMB', 'HML']]

def make_fama_french_mom_df():
  fama_french_mom = load_time_series_csv(FAMA_FRENCH_MOMENTS_CSV, time_format="%Y%m%d")
  return fama_french_mom

if __name__ == '__main__':

  eurostoxx_returns = make_return_df(return_periods=[1, 5, 20])
  riskfree = make_risk_free_df()
  realized_vol = make_realized_vol_df()
  riskneutral_measures = make_riskneutral_df()
  fama_french = make_fama_french_df()
  fama_french_mom = make_fama_french_mom_df()

  df = eurostoxx_returns.join(riskfree, how='inner')
  df = df.join(realized_vol, how='inner')
  df = df.join(riskneutral_measures, how='inner')
  df = df.join(fama_french, how='inner')
  df = df.join(fama_french_mom, how='inner') # add WML (winner-minus-looser) factor
  print(df)