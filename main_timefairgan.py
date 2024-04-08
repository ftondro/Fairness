# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# 1. TimeGAN model
from timegan import timegan
# 2. Data loading
from data_loading import real_data_loading, sine_data_generation, get_original_data
# 3. Metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization


def main (args):
  ## Data loading
  if args.df_name in ['stock', 'energy', 'vital']:
    if args.command == 'no_fairness':
      df, ohe, scaler, discrete_columns_ordereddict, continuous_columns_list, ori_data = real_data_loading(args)
    elif args.command == 'with_fairness':
      df, ohe, scaler, discrete_columns_ordereddict, continuous_columns_list, ori_data, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index = real_data_loading(args)
  elif args.df_name == 'sine':
    # Set number of samples and its dimensions
    no, dim = 10000, 5
    ori_data = sine_data_generation(no, args.seq_len, dim)
 
  print(args.df_name + ' dataset is ready.')
  ## Synthetic data generation by TimeGAN
  # Set newtork parameters
  parameters = dict()
  parameters['command'] = args.command
  parameters['module'] = args.module
  parameters['hidden_dim'] = args.hidden_dim
  parameters['num_layer'] = args.num_layer
  parameters['iterations'] = args.iteration
  parameters['batch_size'] = args.batch_size
  if args.command == 'with_fairness':
    parameters['S'] = args.S
    parameters['Y'] = args.Y
    parameters['S_start_index'] = S_start_index
    parameters['Y_start_index'] = Y_start_index
    parameters['underpriv_index'] = underpriv_index
    parameters['priv_index'] = priv_index
    parameters['undesire_index'] = undesire_index
    parameters['desire_index'] = desire_index
    parameters['underprivileged_value'] = args.underprivileged_value
    parameters['desirable_value'] = args.desirable_value
    parameters['lamda_val'] = args.lamda_val

  generated_data = timegan(ori_data, parameters)   
  print('Finish Synthetic Data Generation')
  synthetic_data_reverse = []
  for row in generated_data:
      for batch in row:
          synthetic_data_reverse.append(list(batch))
  synthetic_data = synthetic_data_reverse[::-1]
  fake_data = get_original_data(synthetic_data, df, ohe, scaler)
  fake_data = fake_data[df.columns]
  fake_data.to_csv('TimeFairGAN' + args.fake_name + '.csv', index = False)
  # Performance metrics   
  # Output initialization
  metric_results = dict()
  # 1. Discriminative Score
  discriminative_score = list()
  for _ in range(args.metric_iteration):
    temp_disc = discriminative_score_metrics(ori_data, generated_data)
    discriminative_score.append(temp_disc)  
  metric_results['discriminative'] = np.mean(discriminative_score)    
  # 2. Predictive score
  predictive_score = list()
  for tt in range(args.metric_iteration):
    temp_pred = predictive_score_metrics(ori_data, generated_data)
    predictive_score.append(temp_pred)       
  metric_results['predictive'] = np.mean(predictive_score) 
  # Print discriminative and predictive scores
  print(metric_results)           
  # 3. Visualization (PCA and tSNE)
  visualization(ori_data, generated_data, 'pca')
  visualization(ori_data, generated_data, 'tsne')
  
  return df, fake_data, metric_results

if __name__ == '__main__':   
  # Inputs for the main function
  parser = argparse.ArgumentParser(description="Script Description")
  subparser = parser.add_subparsers(dest='command')
  with_fairness = subparser.add_parser('with_fairness')
  no_fairness = subparser.add_parser('no_fairness')
  with_fairness.add_argument('--df_name', type=str, help='Dataframe name')
  with_fairness.add_argument('--num_epochs', type=int, help='Number of training epochs')
  with_fairness.add_argument('--batch_size', type=int, help='The batch size')
  with_fairness.add_argument('--seq_len', type=int, help='The sequence length')
  with_fairness.add_argument('--module', type=str, help='gru, lstm, or lstmLN')
  with_fairness.add_argument('--hidden_dim', type=int, help='The hidden dimensions')
  with_fairness.add_argument('--num_layer', type=int, help='Number of layers')
  with_fairness.add_argument('--iteration', type=int, help='Number of training iterations')
  with_fairness.add_argument('--S', type=str, help='Protected attribute', default=None)
  with_fairness.add_argument('--Y', type=str, help='Label (decision)', default=None)
  with_fairness.add_argument('--underprivileged_value', type=str, help='Value for underprivileged group', default=None)
  with_fairness.add_argument('--desirable_value', type=str, help='Desired label (decision)', default=None)
  # parser.add_argument('--num_fair_epochs', type=int, help='Number of fair training epochs', default=None)
  with_fairness.add_argument('--lamda_val', type=float, help='Lambda hyperparameter', default=None)
  with_fairness.add_argument('--metric_iteration', type=int, help='Number of iterations for metric computation')
  with_fairness.add_argument('--fake_name', type=str, help='Name of the produced csv file')
  with_fairness.add_argument('--size_of_fake_data', type=int, help='How many data records to generate')

  no_fairness.add_argument('--df_name', type=str, help='Dataframe name')
  no_fairness.add_argument('--num_epochs', type=int, help='Number of training epochs')
  no_fairness.add_argument('--batch_size', type=int, help='The batch size')
  no_fairness.add_argument('--seq_len', type=int, help='The sequence length')
  no_fairness.add_argument('--module', type=str, help='gru, lstm, or lstmLN')
  no_fairness.add_argument('--hidden_dim', type=int, help='The hidden dimensions')
  no_fairness.add_argument('--num_layer', type=int, help='Number of layers')
  no_fairness.add_argument('--iteration', type=int, help='Number of training iterations')
  no_fairness.add_argument('--metric_iteration', type=int, help='Number of iterations for metric computation')
  no_fairness.add_argument('--fake_name', type=str, help='Name of the produced csv file')
  no_fairness.add_argument('--size_of_fake_data', type=int, help='How many data records to generate')

  args = parser.parse_args()
  ori_data, generated_data, metrics = main(args)
  
