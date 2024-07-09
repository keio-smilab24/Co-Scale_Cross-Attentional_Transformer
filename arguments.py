import sys
import argparse
import datetime

def main_argparse(args=None) -> argparse.ArgumentParser:

    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser(description="command")
    subparsers = parser.add_subparsers(help="sub-command help")

    add_train_subparser(subparsers)
    add_test_subparser(subparsers)

    return parser.parse_args(args)


def add_train_subparser(subparsers: argparse.ArgumentParser) -> None:
    """
    train mode argparse
    """
    parser = subparsers.add_parser("train", help="train mode")
    parser.set_defaults(command="train")

    # path
    parser.add_argument('--checkpointdir', required=True)
    parser.add_argument('--datadir', required=True)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--valid_batch_size', type=int, default=64)
    parser.add_argument('--image', action='store_true', default=False)

    # wandb
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb_name', type=str, default="Co-Scale_Cross-Attentional_Transformer")
    parser.add_argument('--wandb_log_name', type=str, default=f"{datetime.datetime.now()}")


def add_test_subparser(subparsers: argparse.ArgumentParser) -> None:
    """
    test mode argparse
    """
    parser = subparsers.add_parser("test", help="test mode")
    parser.set_defaults(command="test")

    # path
    parser.add_argument('--datadir', required=True)
    parser.add_argument('--checkpointdir', required=True)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--image', action='store_true', default=False)

    # load model
    parser.add_argument('--model', type=str, default='pretrained_model.pth')
