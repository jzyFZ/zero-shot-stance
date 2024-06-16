from datetime import datetime

import torch
from torch import nn
import numpy as np

from utilities.main_arg_parser import job_parser, get_parser
from modeling import data_utils, datasets, model_utils
from train_model import train
from utilities.preparation.ctsan import prepare_ctsan
from utilities.preparation.bicond import prepare_bicond
from utilities.preparation.ffnn_bert import prepare_ffnn_bert
from utilities.preparation.tganet import prepare_tganet
from utilities.preparation.rep_ffnn import prepare_repffnn
from utilities.runtime import print_runtime, print_debug_message

SEED = 0
NUM_GPUS = None
use_cuda = torch.cuda.is_available()


def torch_settings(seed=0):
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)
    torch.backends.cudnn.deterministic = True


def config_settings(args):
    with open(f"config/{args['config_file']}", 'r') as config_file:
        config = dict()
        for line in config_file.readlines():
            config[line.strip().split(":")[0]] = line.strip().split(":")[1]
        return config


def load_vectors(vector_name, vector_dim, seed=0):
    return data_utils.load_vectors('resources/{}.vectors.npy'.format(vector_name), dim=vector_dim, seed=seed)


def run(args, mode, config, vectors, use_cuda, data, dev_dataloader, nn_framework):
    kwargs, batch_args = dict(), dict()
    print_debug_message(f"{__name__}: Running in {mode} mode.")
    if 'BiCond' in mode:
        prepared_model = prepare_bicond(mode=mode, vectors=vectors, use_cuda=use_cuda, config=config)
        name = mode + args['name']
        setup_function = data_utils.setup_helper_bicond
        loss_function = nn.CrossEntropyLoss()
    elif 'CTSAN' in mode:
        prepared_model = prepare_ctsan(mode=mode, vectors=vectors, use_cuda=use_cuda, config=config)
        name = mode + args['name']
        setup_function = data_utils.setup_helper_bicond
        loss_function = nn.CrossEntropyLoss()
    elif 'ffnn-bert' in mode:
        prepared_model = prepare_ffnn_bert(mode=mode, vectors=vectors, use_cuda=use_cuda, config=config)
        name = mode
        batch_args['keep_sen'] = False
        kwargs['fine_tune'] = (config.get('fine-tune', 'no') == 'yes')
        setup_function = data_utils.setup_helper_bert_ffnn
        loss_function = nn.CrossEntropyLoss()
    elif 'tganet' in mode:
        prepared_model = prepare_tganet(mode=mode, vectors=vectors, use_cuda=use_cuda, config=config, nn_framework=nn_framework)
        name = mode
        batch_args['keep_sen'] = False
        setup_function = data_utils.setup_helper_bert_attffnn
        loss_function = nn.CrossEntropyLoss()
    elif 'repffnn' in mode:
        prepared_model = prepare_repffnn(mode=mode, vectors=vectors, use_cuda=use_cuda, config=config)
        name = mode
        batch_args['keep_sen'] = False
        setup_function = data_utils.setup_helper_bert_attffnn
        loss_function = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError()

    kwargs['dataloader'] = data_utils.DataSampler(data, batch_size=int(config['b']))
    kwargs['embed_model'] = prepared_model.get('input_layer')
    kwargs['optimizer'] = prepared_model.get('optimizer')
    kwargs['model'] = prepared_model.get('object_model')
    kwargs['batching_fn'] = data_utils.prepare_batch
    kwargs['loss_function'] = loss_function
    kwargs['batching_kwargs'] = batch_args
    kwargs['setup_fn'] = setup_function
    kwargs['name'] = name

    if nn_framework == "torch":
        model_handler_class = model_utils.TorchModelHandler
    elif nn_framework == "tensorflow":
        model_handler_class = model_utils.TensorFlowModelHandler
    else:
        raise NotImplementedError()

    model_handler = model_handler_class(
        use_cuda=use_cuda,
        # checkpoint_path=config.get('ckp_path', 'data/checkpoints/'),
        result_path=config.get('res_path', 'data/gen-stance/'),
        use_score=args['score_key'],
        save_ckp=args['save_checkpoints'],
        **kwargs
    )

    train(
        model_handler=model_handler,
        num_epochs=int(config['epochs']),
        dev_data=dev_dataloader,
        early_stopping=args['early_stop'],
        num_warm=args['warmup'],
        is_bert=('bert' in config)
    )


def prepare_run(config, args, dev_data_path, use_cuda, seed=0):
    torch_settings(seed=seed)
    train_data_kwargs, dev_data_kwargs = dict(), dict()
    vector_dim = int(config.get('vec_dim', 0))
    vector_name = config.get('vec_name', '')

    if 'topic_name' in config:
        vectors = np.load(
            f"{config['topic_path']}/{config['topic_name']}.{config.get('rep_v', 'centroids')}.npy"
        )
        train_data_kwargs['topic_rep_dict'] = (f"{config['topic_path']}/"
                                               f"{config['topic_name']}-train.labels.pkl")

        dev_data_kwargs['topic_rep_dict'] = (f"{config['topic_path']}/"
                                             f"{config['topic_name']}-dev.labels.pkl")
    elif 'vec_name' in config:
        vectors = load_vectors(vector_name=vector_name, vector_dim=vector_dim, seed=SEED)
    else:
        raise Exception('No vectors provided')

    dev_data = None
    if 'bert' not in config and 'bert' not in config['name']:
        vocab_name = 'resources/{}.vocab.pkl'.format(vector_name)
        data = datasets.StanceData(
            args['train_data'],
            vocab_name,
            pad_val=len(vectors) - 1,
            max_tok_len=int(config.get('max_tok_len', '200')),
            max_sen_len=int(config.get('max_sen_len', '10')),
            keep_sen=('keep_sen' in config),
            **train_data_kwargs
        )
        if args['dev_data'] is not None:
            dev_data = datasets.StanceData(
                dev_data_path,
                vocab_name,
                pad_val=len(vectors) - 1,
                max_tok_len=int(config.get('max_tok_len', '200')),
                max_sen_len=int(config.get('max_sen_len', '10')),
                keep_sen=('keep_sen' in config),
                **dev_data_kwargs)
    else:
        data = datasets.StanceData(
            args['train_data'],
            None,
            max_tok_len=config['max_tok_len'],
            max_top_len=config['max_top_len'],
            is_bert=True,
            add_special_tokens=(config.get('together_in', '0') == '0'),
            **train_data_kwargs
        )
        if dev_data_path is not None:
            dev_data = datasets.StanceData(
                dev_data_path,
                None,
                max_tok_len=config['max_tok_len'],
                max_top_len=config['max_top_len'],
                is_bert=True,
                add_special_tokens=(config.get('together_in', '0') == '0'),
                **dev_data_kwargs)

    dev_dataloader = data_utils.DataSampler(dev_data, batch_size=int(config['b']), shuffle=False) if dev_data else None

    run(
        args=args,
        mode=config['name'],
        config=config,
        vectors=vectors,
        use_cuda=use_cuda,
        data=data,
        dev_dataloader=dev_dataloader,
        nn_framework=args['nn_framework']
    )


if __name__ == '__main__':
    start_time = datetime.now()
    print_debug_message(f"{__name__}: CUDA Availability: {use_cuda}")
    main_parser = job_parser()
    kargs, _ = main_parser.parse_known_args()
    job_mode = vars(kargs)['job']
    main_parser = get_parser(parser=main_parser, job=job_mode)
    kargs, _ = main_parser.parse_known_args()
    args = vars(kargs)
    cs = config_settings(args=args)

    prepare_run(config=cs, args=args, dev_data_path=args['dev_data'], seed=SEED, use_cuda=use_cuda)
    end_time = datetime.now()
    print_runtime(start_time=start_time, end_time=end_time, process_name=cs['name'])
