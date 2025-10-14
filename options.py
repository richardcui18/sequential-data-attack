import argparse
import logging

logger = logging.getLogger(__name__)

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def initialize_parser(self):
        self.parser.add_argument('--dataset', type=str, default='geolife', help="Dataset to use.")
        self.parser.add_argument('--lambda_value', type=float, default=0.1, help="Maximum confidence threshold determined by user.")
        self.parser.add_argument("--deviation_amount_user", type=int, default=0, help="Amount of deviation set by user when generating PI.")
        self.parser.add_argument("--delta", type=float, default=0.7, help="Threshold for reward/penalization.")
        self.parser.add_argument('--num_iter', type=int, default=50, help="Number of iterations in Bi-Directional HMM-RL algorithm.")
        self.parser.add_argument('--k', type=int, default=3, help="Window size in Bi-Directional HMM-RL algorithm.")
        self.parser.add_argument('--gamma', type=int, default=5, help="Gamma in Bi-Directional HMM-RL algorithm.")