# Necessary Packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import QuantileTransformer
from collections import OrderedDict

def sine_data_generation (no, seq_len, dim):
  # Initialize the output
  data = list()
  # Generate sine data
  for i in range(no):      
    # Initialize each time-series
    temp = list()
    # For each feature
    for k in range(dim):
      # Randomly drawn frequency and phase
      freq = np.random.uniform(0, 0.1)            
      phase = np.random.uniform(0, 0.1)       
      # Generate sine signal based on the drawn frequency and phase
      temp_data = [np.sin(freq * j + phase) for j in range(seq_len)] 
      temp.append(temp_data)   
    # Align row/column
    temp = np.transpose(np.asarray(temp))        
    # Normalize to [0,1]
    temp = (temp + 1)*0.5
    # Stack the generated data
    data.append(temp)             
  return data
    
def get_ohe_data(df):
  df_int = df.select_dtypes(['float', 'integer']).values
  continuous_columns_list = list(df.select_dtypes(['float', 'integer']).columns)
  scaler = QuantileTransformer(n_quantiles=2000, output_distribution='uniform')
  df_int = scaler.fit_transform(df_int)
  df_cat = df.select_dtypes('object')
  df_cat_names = list(df.select_dtypes('object').columns)
  numerical_array = df_int
  ohe = OneHotEncoder()
  ohe_array = ohe.fit_transform(df_cat)
  cat_lens = [i.shape[0] for i in ohe.categories_]
  discrete_columns_ordereddict = OrderedDict(zip(df_cat_names, cat_lens))
  final_array = np.hstack((numerical_array, ohe_array.toarray()))
  return df, ohe, scaler, discrete_columns_ordereddict, continuous_columns_list, final_array

def get_ohe_data_fair(df, S_under, Y_desire, S, Y):
  df_int = df.select_dtypes(['float', 'integer']).values
  continuous_columns_list = list(df.select_dtypes(['float', 'integer']).columns)
  scaler = QuantileTransformer(n_quantiles=2000, output_distribution='uniform')
  df_int = scaler.fit_transform(df_int)
  df_cat = df.select_dtypes('object')
  df_cat_names = list(df.select_dtypes('object').columns)
  numerical_array = df_int
  ohe = OneHotEncoder()
  ohe_array = ohe.fit_transform(df_cat)
  cat_lens = [i.shape[0] for i in ohe.categories_]
  discrete_columns_ordereddict = OrderedDict(zip(df_cat_names, cat_lens))
  S_start_index = len(continuous_columns_list) + sum(
                  list(discrete_columns_ordereddict.values())[:list(discrete_columns_ordereddict.keys()).index(S)])
  Y_start_index = len(continuous_columns_list) + sum(
                  list(discrete_columns_ordereddict.values())[:list(discrete_columns_ordereddict.keys()).index(Y)])
  if ohe.categories_[list(discrete_columns_ordereddict.keys()).index(S)][0] == S_under:
      underpriv_index = 0
      priv_index = 1
  else:
      underpriv_index = 1
      priv_index = 0
  if ohe.categories_[list(discrete_columns_ordereddict.keys()).index(Y)][0] == Y_desire:
      desire_index = 0
      undesire_index = 1
  else:
      desire_index = 1
      undesire_index = 0
  final_array = np.hstack((numerical_array, ohe_array.toarray()))
  return df, ohe, scaler, discrete_columns_ordereddict, continuous_columns_list, final_array, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index

def real_data_loading (args):
  assert args.df_name in ['stock','energy', 'vital']
  if args.df_name == 'stock':
    df = pd.read_csv('data/stock_data.csv')
  elif args.df_name == 'energy':
    df = pd.read_csv('data/energy_data.csv')
  elif args.df_name == 'vital':
    df = pd.read_csv('data/vital_signs_data.csv')
  if args.command == 'with_fairness':
    S = args.S
    Y = args.Y
    S_under = args.underprivileged_value
    Y_desire = args.desirable_value
    df[S] = df[S].astype(object)
    df[Y] = df[Y].astype(object)
    df, ohe, scaler, discrete_columns_ordereddict, continuous_columns_list, df_transformed, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index = get_ohe_data_fair(df, S_under, Y_desire, S, Y)
  elif args.command == 'no_fairness':
    df, ohe, scaler, discrete_columns_ordereddict, continuous_columns_list, df_transformed = get_ohe_data(df)
  ori_data = df_transformed  
  # Flip the data to make chronological data
  ori_data = ori_data[::-1]
  # Preprocess the dataset
  temp_data = []    
  # Cut data by sequence length
  for i in range(0, len(ori_data) - args.seq_len):
    _x = ori_data[i:i + args.seq_len]
    temp_data.append(_x)     
  # Mix the datasets (to make it similar to i.i.d)
  idx = np.random.permutation(len(temp_data))    
  data = []
  for i in range(len(temp_data)):
    data.append(temp_data[idx[i]])
  if args.command == 'no_fairness': 
    return df, ohe, scaler, discrete_columns_ordereddict, continuous_columns_list, data
  elif args.command == 'with_fairness': 
    return df, ohe, scaler, discrete_columns_ordereddict, continuous_columns_list, data, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index

def get_original_data(df_transformed, df_orig, ohe, scaler):
  df_transformed = np.array(df_transformed)
  df_ohe_int = df_transformed[:,:df_orig.select_dtypes(['float', 'integer']).shape[1]]
  df_ohe_int = scaler.inverse_transform(df_ohe_int)
  df_ohe_cats = df_transformed[:, df_orig.select_dtypes(['float', 'integer']).shape[1]:]
  df_ohe_cats = ohe.inverse_transform(df_ohe_cats)
  df_int = pd.DataFrame(df_ohe_int, columns=df_orig.select_dtypes(['float', 'integer']).columns)
  df_cat = pd.DataFrame(df_ohe_cats, columns=df_orig.select_dtypes('object').columns)
  return pd.concat([df_int, df_cat], axis=1)