import argparse
import os
import math
import tensorforce
import hparams
from environment import SEDEnvironment
from util import load_data


def parse_args():
  parser = argparse.ArgumentParser(description='SED_RL')
  parser.add_argument('--path_to_data', type=str, required=True,
                      help='Path to .npy files containing event probabilities.')
  parser.add_argument('--path_to_reference_data', type=str, required=True,
                      help='Path to .txt files containing reference event labels.')
  parser.add_argument('--output_dir', type=str, required=True,
                      help='Path to output directory.')
  parser.add_argument('--name', type=str, required=True,
                      help='Name of the experiment run.')
  parser.add_argument('--mode', choices=['train', 'eval'], default='train',
                      help='Train or evaluate the agent.')
  args = parser.parse_args()
  return args


if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  args = parse_args()

  # Load data and reference_data
  max_length = math.ceil(hparams.max_length_sec * (hparams.sample_rate / hparams.hop_length))
  data, reference_data = load_data(args.path_to_data, args.path_to_reference_data, max_length=max_length)

  # Create environment
  sed_environment = SEDEnvironment(data=data, reference_data=reference_data, max_length=max_length)
  environment = tensorforce.Environment.create(environment=sed_environment)

  # Create agent network
  network = [
    dict(type='input_gru', size=hparams.state_size, return_final_state=False),
    dict(type='input_gru', size=hparams.state_size, return_final_state=False),
    dict(type='pooling', reduction='mean')
  ]

  logs_dir = os.path.join(args.output_dir, 'logs', args.name)
  checkpoint_dir = os.path.join(args.output_dir, 'checkpoints', args.name)

  agent = None
  if args.mode == 'train':
    # Create the agent
    summarizer = dict(
      directory=logs_dir,
      summaries=['reward', 'entropy', 'loss']
    )
    saver = dict(
      directory=checkpoint_dir,
      frequency=10
    )
    agent = tensorforce.Agent.create(agent='vpg',
                                     environment=environment,
                                     network=network,
                                     batch_size=hparams.vpg_batch_size,
                                     memory=hparams.vpg_memory,
                                     learning_rate=hparams.vpg_learning_rate,
                                     update_frequency=hparams.vpg_update_frequency,
                                     discount=hparams.vpg_discount,
                                     entropy_regularization=hparams.vpg_entropy_regularization,
                                     exploration=hparams.vpg_exploration,
                                     state_preprocessing=None,
                                     summarizer=summarizer,
                                     saver=saver)

  else:
    agent = tensorforce.Agent.load(directory=checkpoint_dir)
    agent.spec['exploration'] = 0.0

  # Create runner
  runner = tensorforce.Runner(agent=agent, environment=environment)
  # Train
  runner.run(num_episodes=hparams.num_episodes, evaluation=True if args.mode=='eval' else False)
  # Close environment and agent
  environment.close()
  agent.close()