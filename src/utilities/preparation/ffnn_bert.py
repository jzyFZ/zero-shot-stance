from torch import optim

from modeling import input_models, models, input_models_tf, models_tf


def prepare_ffnn_bert(mode, vectors, use_cuda, config, nn_framework):
    if nn_framework == 'torch':
        input_layer_class = input_models.JointBERTLayerWithExtra
        object_model_class = models.TGANet
    elif nn_framework == 'tensorflow':
        input_layer_class = input_models_tf.JointBERTLayerWithExtraTF
        object_model_class = models_tf.TGANetTF
    else:
        raise NotImplementedError('Unknown NN framework: {}'.format(nn_framework))

    if config.get('together_in', '0') == '1':
        if 'topic_name' in config:
            input_layer = input_models.JointBERTLayerWithExtra(
                vecs=vectors,
                use_cuda=use_cuda,
                use_both=(config.get('use_ori_topic', '1') == '1'),
                static_vecs=(config.get('static_topics', '1') == '1')
            )
        else:
            input_layer = input_models.JointBERTLayer(use_cuda=use_cuda)
    else:
        input_layer = input_models.BERTLayer(mode='text-level', use_cuda=use_cuda)

    object_model = models.FFNN(
        input_dim=input_layer.dim,
        in_dropout_prob=float(config['in_dropout']),
        hidden_size=int(config['hidden_size']),
        add_topic=(config.get('add_resid', '1') == '1'),
        use_cuda=use_cuda,
        bias=False
    )

    optimizer = optim.Adam(object_model.parameters())

    return {
        'object_model': object_model,
        'input_layer': input_layer,
        'optimizer': optimizer,
    }
