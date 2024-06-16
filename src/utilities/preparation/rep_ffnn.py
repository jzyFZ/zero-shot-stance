from torch import optim

from modeling import input_models, models, input_models_tf, models_tf


def prepare_repffnn(mode, vectors, use_cuda, config, nn_framework):
    if nn_framework == 'torch':
        input_layer_class = input_models.JointBERTLayerWithExtra
        object_model_class = models.RepFFNN
    elif nn_framework == 'tensorflow':
        input_layer_class = input_models_tf.JointBERTLayerWithExtraTF
        object_model_class = models_tf.RepFFNNTF
    else:
        raise NotImplementedError('Unknown NN framework: {}'.format(nn_framework))

    input_layer = input_layer_class(
        vecs=vectors,
        use_cuda=use_cuda,
        use_both=(config.get('use_ori_topic', '1') == '1'),
        static_vecs=(config.get('static_topics', '1') == '1')
    )

    object_model = object_model_class(
        in_dropout_prob=float(config['in_dropout']),
        hidden_size=int(config['hidden_size']),
        input_dim=int(config['topic_dim']),
        use_cuda=use_cuda
    )

    optimizer = optim.Adam(object_model.parameters())

    return {
        'object_model': object_model,
        'input_layer': input_layer,
        'optimizer': optimizer,
    }
