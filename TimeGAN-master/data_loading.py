"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

data_loading.py

(0) MinMaxScaler: Min Max normalizer
(1) sine_data_generation: Generate sine dataset
(2) real_data_loading: Load and preprocess real data
  - stock_data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG
  - energy_data: http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
"""

## Necessary Packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import QuantileTransformer
from collections import OrderedDict

def MinMaxScaler(data):
  """Min Max normalizer.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
  """
  numerator = data - np.min(data, 0)
  denominator = np.max(data, 0) - np.min(data, 0)
  norm_data = numerator / (denominator + 1e-7)
  return norm_data


def sine_data_generation (no, seq_len, dim):
  """Sine data generation.
  
  Args:
    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions
    
  Returns:
    - data: generated data
  """  
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
  return ohe, scaler, discrete_columns_ordereddict, continuous_columns_list, final_array

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
  return ohe, scaler, discrete_columns_ordereddict, continuous_columns_list, final_array, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index

def real_data_loading (args):
  """Load and preprocess real-world datasets.
  
  Args:
    - data_name: stock or energy
    - seq_len: sequence length
    
  Returns:
    - data: preprocessed data.
  """  
  assert args.df_name in ['stock','energy', 'vital']
  
  if args.df_name == 'stock':
    df = pd.read_csv('data/stock_data.csv')
    #ori_data = np.loadtxt('data/stock_data.csv', delimiter = ",",skiprows = 1)
  elif args.df_name == 'energy':
    df = pd.read_csv('data/energy_data.csv')
    # ori_data = np.loadtxt('data/energy_data.csv', delimiter = ",",skiprows = 1)
  elif args.df_name == 'vital':
    df = pd.read_csv('data/vital_signs_data.csv')
  if args.command == 'with_fairness':
    S = args.S
    Y = args.Y
    S_under = args.underprivileged_value
    Y_desire = args.desirable_value
    df[S] = df[S].astype(object)
    df[Y] = df[Y].astype(object)
    ohe, scaler, discrete_columns_ordereddict, continuous_columns_list, df_transformed, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index = get_ohe_data_fair(df, S_under, Y_desire, S, Y)
  elif args.command == 'no_fairness':
    ohe, scaler, discrete_columns_ordereddict, continuous_columns_list, df_transformed = get_ohe_data(df)
  ori_data = df_transformed     
  # Flip the data to make chronological data
  ori_data = ori_data[::-1]
  # Normalize the data
  #ori_data = MinMaxScaler(ori_data)
  
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
    return ohe, scaler, discrete_columns_ordereddict, continuous_columns_list, data
  elif args.command == 'with_fairness':
    return ohe, scaler, discrete_columns_ordereddict, continuous_columns_list, data, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index