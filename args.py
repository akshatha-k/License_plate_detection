import argparse

parser = argparse.ArgumentParser(description='LP Detection')
# Training Parameters
parser.add_argument('--model', type=str, default='eccv', help='Path to previous model')
parser.add_argument('--name', type=str, default='eccv', help='Model name')
parser.add_argument('--lp_model', type=str, default=None,
                    help='Pre-trained model path')

# training params
parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer (default = Adam)')
parser.add_argument('--learning_rate', type=float, default=.01, help='Optimizer (default = 0.01)')
parser.add_argument('--batch_size', type=int, default=32, help='Mini-batch size (default = 32)')
parser.add_argument('--image_size', type=int, default=208, help='Image size')
parser.add_argument(
    '--epochs', default=10, type=int, help='Number of training epochs')
parser.add_argument(
    '--num_augs', default=1000, type=int, help='Total number of images after random augmentations')

# colab specific
parser.add_argument(
    '--use_colab', default=True, type=bool, help='Use google colab')
parser.add_argument(
    '--resume', default=False, type=bool, help='Resume from ckpt')

# prune params
parser.add_argument(
    '--prune_model', default=False, type=bool, help='Whether to prune the model or not')
parser.add_argument(
    '--initial_sparsity', type=float, default=0.0, help='Initial sparsity while pruning')
parser.add_argument(
    '--final_sparsity', type=float, default=0.5, help='Final sparsity while pruning')
parser.add_argument(
    '--begin_step', type=int, default=0, help='Start pruning point')
parser.add_argument(
    '--end_step', type=int, default=200, help='End pruning point')

# cyclical learning rate params
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
