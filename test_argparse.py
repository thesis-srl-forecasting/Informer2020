import argparse

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
args = parser.parse_args()

print(args)

args.learning_rate = 0.1
print(args)