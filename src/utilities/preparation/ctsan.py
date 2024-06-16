from torch import optim

from modeling import input_models, models, input_models_tf, models_tf


def prepare_ctsan(mode, vectors, use_cuda, config, nn_framework):
    if nn_framework == 'torch':
        input_layer_class = input_models.BasicWordEmbedLayer
        object_model_class = models.CTSAN
    elif nn_framework == 'tensorflow':
        input_layer_class = input_models_tf.BasicWordEmbedLayerTF
        object_model_class = models_tf.CTSANTF
    else:
        raise NotImplementedError('Unknown NN framework: {}'.format(nn_framework))

    input_layer = input_layer_class(vecs=vectors, use_cuda=use_cuda)

    object_model = object_model_class(
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

    lr = float(config.get('lr', '0.001'))
    optimizer = optim.Adam(object_model.parameters(), lr=lr)

    return {
        'object_model': object_model,
        'input_layer': input_layer,
        'optimizer': optimizer,
    }
