import argparse

parser = argparse.ArgumentParser(description='LP Detection')
# Training Parameters
parser.add_argument('--model'			,type=str   , default='eccv'		,help='Path to previous model')
parser.add_argument('--name'			,type=str   , default='eccv', help='Model name')
parser.add_argument('--model_dir'		,type=str   , default='/content/License_plate_detection/models/eccv-model-scracth', help='Save models')
parser.add_argument('--train_dir'		,type=str   , default='', help='Input data directory for training')
parser.add_argument('--iterations'		,type=int   , default=300000	,help='Number of mini-batch iterations (default = 300.000)')
parser.add_argument('--batch_size'		,type=int   , default=32		,help='Mini-batch size (default = 32)')
parser.add_argument('--input_dir'		,type=str   , default='/content/License_plate_detection/samples/test'	,help='Input directory (default = ./)')
parser.add_argument('--lp_model'		,type=str   , default='data/lp-detector/wpod-net_update1.h5', help='Pre-trained model path')
parser.add_argument('--output_dir'		,type=str   , default='/content/License_plate_detection/rand/output'	,help='Output directory (default = ./)')
parser.add_argument('--optimizer'		,type=str   , default='Adam'	,help='Optmizer (default = Adam)')
parser.add_argument('--learning_rate'	,type=float , default=.01		,help='Optmizer (default = 0.01)')

parser.add_argument(
  '--prune_model', default=False, type=bool, help='Whether to prune the model or not')
parser.add_argument(
  '--epochs', default=10, type=int, help='Number of training epochs')
parser.add_argument(
  '--use_colab', default=True, type=bool, help='Use google colab')
parser.add_argument(
  '--resume', default=False, type=bool, help='Resume from ckpt')


#cyclical learning rate params
parser.add_argument(
  '--lr_steps', default=5000, type=int, help='Cycle step for cyclical LR')
parser.add_argument(
  '--lr_schedule', default='cyclic', type=str, help='LR scheduler to use cyclic,step')
parser.add_argument(
  '--max_lr', default=0.1, type=float, help='Max Learning rate')
parser.add_argument(
  '--min_lr', default=0, type=float, help='Min Learning rate')


def get_args():
  args = parser.parse_args()

  return args
