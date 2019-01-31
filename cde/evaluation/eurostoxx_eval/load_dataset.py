import numpy as np
import os
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline

from cde.utils.io import load_time_series_csv



DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data'))

EURO_OIS_CSV = os.path.join(DATA_DIR, "2_Eurostoxx50/eur_ois.csv")
EUROSTOXX_CSV = os.path.join(DATA_DIR, "2_Eurostoxx50/eurostoxx50_prices_eod.csv")
EURO_TAIL_VARIATION_CSV = os.path.join(DATA_DIR, "2_Eurostoxx50/eurostoxx50_exp_tail_variation_measures.csv")
REALIZED_VOL_CSV = os.path.join(DATA_DIR, "2_Eurostoxx50/eurostoxx50_realized_volmeasures.csv")
RISKNEUTRAL_CSV = os.path.join(DATA_DIR,"2_Eurostoxx50/eurostoxx50_riskneutralmeasures.csv")
VRP_CSV = os.path.join(DATA_DIR, "2_Eurostoxx50/eurostoxx50_vrp.csv")
FAMA_FRENCH_CSV = os.path.join(DATA_DIR, "2_Eurostoxx50/FamaFrench_Europe_3_Factors_Daily.csv")
FAMA_FRENCH_MOMENTUM_CSV = os.path.join(DATA_DIR, "2_Eurostoxx50/FamaFrench_Europe_MOM_Factor_Daily.csv")


""" HELPER METHODS """

def _make_return_df(return_periods):
  eurostoxx = load_time_series_csv(EUROSTOXX_CSV)
  for h in return_periods:
    eurostoxx['log_ret_%i'%h] = np.log(eurostoxx.lastprice) - np.log(eurostoxx.lastprice.shift(h))

  # compute last period return
  eurostoxx['log_ret_last_period'] = (np.log(eurostoxx.lastprice) - np.log(eurostoxx.lastprice.shift(1))).shift(1)
  return eurostoxx.drop(labels=['lastprice'], axis=1)

def _make_risk_free_df():
  euro_oid = load_time_series_csv(EURO_OIS_CSV)
  euro_oid = euro_oid[euro_oid.maturity == 1]
  euro_oid['log_risk_free_1d'] = np.log((euro_oid['yield']/365) + 1)
  return euro_oid.drop(labels=['maturity', 'yield'], axis=1)

def _make_exp_tail_variation_df():
  return load_time_series_csv(EURO_TAIL_VARIATION_CSV)

def _make_realized_vol_df():
  realized_vol = load_time_series_csv(REALIZED_VOL_CSV)
  return realized_vol.loc[:, ['RealizedVariation']]

def _make_riskneutral_df(time_horizon):
  cols_of_interest = ['bakshiSkew', 'bakshiKurt', 'SVIX',]
  riskteural_measures = load_time_series_csv(RISKNEUTRAL_CSV, delimiter=';')
  riskteural_measures = riskteural_measures[['daystomaturity'] + cols_of_interest]
  riskteural_measures = riskteural_measures.dropna()
  interpolated_df = pd.DataFrame()
  for date in list(set(riskteural_measures.index)):
    # filter all row for respective date
    riskneutral_measures_per_day = riskteural_measures.ix[date]

    # filer out all option-implied measures with computed based on a maturity of less than 7 days
    riskneutral_measures_per_day = riskneutral_measures_per_day[riskneutral_measures_per_day['daystomaturity'] > 7]

    # interpolate / extrapolate to get estimate for desired time_horizon
    interpolated_values = [InterpolatedUnivariateSpline(np.array(riskneutral_measures_per_day['daystomaturity']),
                                 np.asarray(riskneutral_measures_per_day[col_of_interest]),
                                 k=1)(time_horizon) for col_of_interest in cols_of_interest]

    # create df with estimated option-implied risk measures
    update_dict = dict(zip(cols_of_interest, interpolated_values))
    update_dict.update({'daystomaturity': time_horizon})
    interpolated_df = interpolated_df.append(pd.DataFrame(update_dict, index=[date]))
  del interpolated_df['daystomaturity']
  return interpolated_df

def _make_variance_risk_premium_df():
  return load_time_series_csv(VRP_CSV, delimiter=';')

def _make_fama_french_df():
  fama_french_factors = load_time_series_csv(FAMA_FRENCH_CSV, time_format="%Y%m%d")
  return fama_french_factors.loc[:, ['Mkt-RF', 'SMB', 'HML']]

def _make_fama_french_mom_df():
  return load_time_series_csv(FAMA_FRENCH_MOMENTUM_CSV, time_format="%Y%m%d")

def _compute_frama_french_factor_risk(df, time_steps):
  assert set(['WML', "Mkt-RF", 'SMB', 'HML']) <= set(df.columns)
  for ts in time_steps:
    for factor in ['WML', "Mkt-RF", 'SMB', 'HML']:
      df[factor + '_risk_%id'%ts] = df[factor].rolling(ts).sum()
  return df

""" PUBLIC METHODS """

def make_overall_eurostoxx_df(return_period=1):
  eurostoxx_returns = _make_return_df(return_periods=[return_period])
  riskfree = _make_risk_free_df()
  realized_vol = _make_realized_vol_df()
  riskneutral_measures = _make_riskneutral_df(time_horizon=30)
  fama_french = _make_fama_french_df()
  fama_french_mom = _make_fama_french_mom_df()

  df = eurostoxx_returns.join(riskfree, how='inner')
  df = df.join(realized_vol, how='inner')
  df = df.join(riskneutral_measures, how='inner')
  df = df.join(fama_french, how='inner')
  df = df.join(fama_french_mom, how='inner')  # add WML (winner-minus-looser) factor
  df = _compute_frama_french_factor_risk(df, [10])
  return df

def target_feature_split(df, target_col, filter_nan=True, return_features=False):
  assert target_col in df.columns

  if filter_nan:
    nrows = df.shape[0]
    df = df.dropna()
    print("Dropping %i rows from frame"%(nrows-df.shape[0]))

  Y = np.array(df[target_col])
  X = np.array(df.loc[:, df.columns != target_col])

  assert X.shape[0] == Y.shape[0]
  if return_features:
    features = df.columns[df.columns != target_col]
    return X, Y, features
  else:
     return X, Y

if __name__ == '__main__':
  df = make_overall_eurostoxx_df()

  X, Y = target_feature_split(df, 'log_ret_1', filter_nan=True)

  print(X, Y)