"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

main_timegan.py

(1) Import data
(2) Generate synthetic data
(3) Evaluate the performances in three ways
  - Visualization (t-SNE, PCA)
  - Discriminative score
  - Predictive score
"""

## Necessary packages
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
from data_loading import real_data_loading, sine_data_generation
# 3. Metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization


def main (args):
  """Main function for timeGAN experiments.
  
  Args:
    - data_name: sine, stock, or energy
    - seq_len: sequence length
    - Network parameters (should be optimized for different datasets)
      - module: gru, lstm, or lstmLN
      - hidden_dim: hidden dimensions
      - num_layer: number of layers
      - iteration: number of training iterations
      - batch_size: the number of samples in each batch
    - metric_iteration: number of iterations for metric computation
  
  Returns:
    - ori_data: original data
    - generated_data: generated synthetic data
    - metric_results: discriminative and predictive scores
  """
  ## Data loading
  if args.df_name in ['stock', 'energy', 'vital']:
    if args.command == 'no_fairness':
      ohe, scaler, discrete_columns_ordereddict, continuous_columns_list, ori_data = real_data_loading(args)
    elif args.command == 'with_fairness':
      ohe, scaler, discrete_columns_ordereddict, continuous_columns_list, ori_data, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index = real_data_loading(args)
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
  
  ## Performance metrics   
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
          
  # 3. Visualization (PCA and tSNE)
  visualization(ori_data, generated_data, 'pca')
  visualization(ori_data, generated_data, 'tsne')
  
  ## Print discriminative and predictive scores
  print(metric_results)

  return ori_data, generated_data, metric_results


if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser(description="Script Description")
  subparser = parser.add_subparsers(dest='command')
  with_fairness = subparser.add_parser('with_fairness')
  no_fairness = subparser.add_parser('no_fairness')
  # Adding all arguments upfront, including those conditionally used
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
  # Calls main function  
  ori_data, generated_data, metrics = main(args)