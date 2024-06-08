import argparse


def job_parser():
    parser = argparse.ArgumentParser(description='Zero Shot Stance Model')
    parser.add_argument('-j', '--job', choices=[
        'train', 'eval', 'hyper-param'
    ], help='Select corresponding job (train, eval, hyper-param)', required=True)
    return parser


def get_parser(parser=argparse.ArgumentParser(), job='train') -> argparse:
    parser.add_argument(
        '-c', '--config-file', help='Name of the config data file', required=True
    )
    parser.add_argument(
        '-t', '--train-data', help='Name of the training data file', required=True
    )
    parser.add_argument(
        '-d', '--dev-data', help='Name of the dev data file', default=None, required=False
    )
    parser.add_argument(
        '-n', '--name', help='Something to add to the saved model name',
        required=False, default=''
    )
    parser.add_argument(
        '-o', '--out', help='Output file name', default=''
    )
    parser.add_argument(
        '-e', '--early-stop', help='Whether to do early stopping or not',
        required=False, type=bool, default=False
    )
    parser.add_argument(
        '-w', '--warmup', help='Number of warm-up epochs', required=False,
        type=int, default=0
    )
    parser.add_argument(
        '-k', '--score-key', help='Score to use for optimization', required=False,
        default='f_macro'
    )
    parser.add_argument(
        '-v', '--verbose', help='Verbose mode', required=False,
        default=False, type=bool
    )
    if job == 'train':
        parser.add_argument(
            '-s', '--save-checkpoints', help='Whether to save checkpoints',
            required=False, default=False, type=bool
        )
    elif job == 'eval':
        parser.add_argument(
            '-p', '--checkpoint-name', help='Checkpoint name', required=False
        )
        parser.add_argument(
            '-m', '--mode', help='What to do', required=True
        )

    return parser
