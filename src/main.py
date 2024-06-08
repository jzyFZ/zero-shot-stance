import time

import torch
from torch import nn, optim
import numpy as np

from main_arg_parser import job_parser, get_parser
from modeling import models, data_utils, datasets, input_models, model_utils
from train_model import train

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


def run(args, mode, config, vectors, use_cuda, data, dev_dataloader):
    lr = float(config.get('lr', '0.001'))  # set the optimizer

    kwargs, batch_args, name = dict(), dict(), mode  # config['name']
    if 'BiCond' in mode:
        name += args['name']
        input_layer = input_models.BasicWordEmbedLayer(
            vecs=vectors,
            use_cuda=use_cuda,
            static_embeds=(config.get('tune_embeds', '0') == '0')
        )
        setup_function = data_utils.setup_helper_bicond
        loss_function = nn.CrossEntropyLoss()

        object_model = models.BiCondLSTMModel(
            hidden_dim=int(config['h']),
            embed_dim=input_layer.dim,
            input_dim=(int(config['in_dim']) if 'in_dim' in config['name'] else input_layer.dim),
            drop_prob=float(config['dropout']), use_cuda=use_cuda,
            num_labels=3,
            keep_sentences=('keep_sen' in config),
            doc_method=config.get('doc_m', 'maxpool')
        )
        optimizer = optim.Adam(object_model.parameters(), lr=lr)
    elif 'CTSAN' in mode:
        name += args['name']
        input_layer = input_models.BasicWordEmbedLayer(vecs=vectors, use_cuda=use_cuda)
        setup_function = data_utils.setup_helper_bicond
        loss_function = nn.CrossEntropyLoss()

        object_model = models.CTSAN(
            hidden_dim=int(config['h']),
            embed_dim=input_layer.dim,
            att_dim=int(config['a']),
            lin_size=int(config['lh']),
            drop_prob=float(config['dropout']),
            use_cuda=use_cuda,
            out_dim=3,
            keep_sentences=('keep_sen' in config),
            sentence_version=config.get('sen_v', 'default'),
            doc_method=config.get('doc_m', 'maxpool'),
            premade_topic=('topic_name' in config),
            topic_trans=('topic_name' in config),
            topic_dim=(int(config.get('topic_dim')) if 'topic_dim' in config else None)
        )
        optimizer = optim.Adam(object_model.parameters(), lr=lr)
    elif 'ffnn-bert' in mode:
        batch_args['keep_sen'] = False
        if config.get('together_in', '0') == '1':
            if 'topic_name' in config:
                input_layer = input_models.JointBERTLayerWithExtra(
                    vecs=vectors,
                    use_cuda=use_cuda,
                    use_both=(config.get('use_ori_topic', '1') == '1'),
                    static_vecs=(config.get('static_topics', '1') == '1'))
            else:
                input_layer = input_models.JointBERTLayer(use_cuda=use_cuda)
        else:
            input_layer = input_models.BERTLayer(mode='text-level', use_cuda=use_cuda)

        setup_function = data_utils.setup_helper_bert_ffnn

        loss_function = nn.CrossEntropyLoss()
        object_model = models.FFNN(
            input_dim=input_layer.dim,
            in_dropout_prob=float(config['in_dropout']),
            hidden_size=int(config['hidden_size']),
            add_topic=(config.get('add_resid', '1') == '1'),
            use_cuda=use_cuda,
            bias=False
        )
        optimizer = optim.Adam(object_model.parameters())

        kwargs['fine_tune'] = (config.get('fine-tune', 'no') == 'yes')
    elif 'tganet' in mode:
        batch_args['keep_sen'] = False
        input_layer = input_models.JointBERTLayerWithExtra(
            vecs=vectors,
            use_cuda=use_cuda,
            use_both=(config.get('use_ori_topic', '1') == '1'),
            static_vecs=(config.get('static_topics', '1') == '1')
        )

        setup_function = data_utils.setup_helper_bert_attffnn

        loss_function = nn.CrossEntropyLoss()

        object_model = models.TGANet(
            in_dropout_prob=float(config['in_dropout']),
            hidden_size=int(config['hidden_size']),
            text_dim=int(config['text_dim']),
            add_topic=(config.get('add_resid', '0') == '1'),
            att_mode=config.get('att_mode', 'text_only'),
            topic_dim=int(config['topic_dim']),
            learned=(config.get('learned', '0') == '1'),
            use_cuda=use_cuda
        )

        optimizer = optim.Adam(object_model.parameters())
    elif 'repffnn' in mode:
        batch_args['keep_sen'] = False
        input_layer = input_models.JointBERTLayerWithExtra(
            vecs=vectors,
            use_cuda=use_cuda,
            use_both=(config.get('use_ori_topic', '1') == '1'),
            static_vecs=(config.get('static_topics', '1') == '1')
        )

        setup_function = data_utils.setup_helper_bert_attffnn

        loss_function = nn.CrossEntropyLoss()

        object_model = models.RepFFNN(
            in_dropout_prob=float(config['in_dropout']),
            hidden_size=int(config['hidden_size']),
            input_dim=int(config['topic_dim']),
            use_cuda=use_cuda
        )

        optimizer = optim.Adam(object_model.parameters())
    else:
        raise NotImplementedError()

    kwargs['dataloader'] = data_utils.DataSampler(data, batch_size=int(config['b']))
    kwargs['model'] = object_model
    kwargs['embed_model'] = input_layer
    kwargs['batching_fn'] = data_utils.prepare_batch
    kwargs['batching_kwargs'] = batch_args
    kwargs['name'] = name
    kwargs['loss_function'] = loss_function
    kwargs['optimizer'] = optimizer
    kwargs['setup_fn'] = setup_function

    model_handler = model_utils.TorchModelHandler(
        use_cuda=use_cuda,
        checkpoint_path=config.get('ckp_path', 'data/checkpoints/'),
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
    )


if __name__ == '__main__':
    start_time = time.time()
    print("CUDA Availability:", use_cuda)
    main_parser = job_parser()
    kargs, _ = main_parser.parse_known_args()
    job_mode = vars(kargs)['job']
    main_parser = get_parser(parser=main_parser, job=job_mode)
    kargs, _ = main_parser.parse_known_args()
    args = vars(kargs)
    cs = config_settings(args=args)

    prepare_run(config=cs, args=args, dev_data_path=args['dev_data'], seed=SEED, use_cuda=use_cuda)

    print("[{}] total runtime: {:.2f} minutes".format(cs['name'], (time.time() - start_time) / 60.))
