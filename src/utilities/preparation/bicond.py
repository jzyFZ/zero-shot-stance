from torch import optim
import tensorflow as tf
from modeling import input_models, models, input_models_tf, models_tf


def prepare_bicond(mode, vectors, use_cuda, config, nn_framework):
    lr = float(config.get('lr', '0.001'))

    if nn_framework == 'torch':
        input_layer_class = input_models.BasicWordEmbedLayer
        object_model_class = models.BiCondLSTMModel
    elif nn_framework == 'tensorflow':
        input_layer_class = input_models_tf.BasicWordEmbedLayerTF
        object_model_class = models_tf.BiCondLSTMModelTF
    else:
        raise NotImplementedError('Unknown NN framework: {}'.format(nn_framework))

    input_layer = input_layer_class(
        vecs=vectors,
        use_cuda=use_cuda,
        static_embeds=(config.get('tune_embeds', '0') == '0')
    )

    object_model = object_model_class(
        hidden_dim=int(config['h']),
        embed_dim=input_layer.dim,
        input_dim=(int(config['in_dim']) if 'in_dim' in mode else input_layer.dim),
        drop_prob=float(config['dropout']), use_cuda=use_cuda,
        num_labels=3,
        keep_sentences=('keep_sen' in config),
        doc_method=config.get('doc_m', 'maxpool')
    )

    if nn_framework == 'torch':
        optimizer = optim.Adam(object_model.parameters(), lr=lr)
    elif nn_framework == 'tensorflow':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        raise NotImplementedError('Unknown NN framework: {}'.format(nn_framework))

    return {
        'object_model': object_model,
        'input_layer': input_layer,
        'optimizer': optimizer,
    }
