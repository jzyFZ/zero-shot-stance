import tensorflow as tf
from tensorflow.keras import layers
from modeling import model_layers_tf


class FeedForwardNeuralNetworkTF(tf.keras.Model):
    def __init__(self, **kwargs):
        super(FeedForwardNeuralNetworkTF, self).__init__()
        self.use_cuda = kwargs['use_cuda']
        self.num_labels = kwargs.get('num_labels', 3)
        self.use_topic = kwargs['add_topic']

        if 'input_dim' in kwargs:
            if self.use_topic:
                in_dim = 2 * kwargs['input_dim']
            else:
                in_dim = kwargs['input_dim']
        else:
            in_dim = kwargs['topic_dim'] + kwargs['text_dim']

        self.model = tf.keras.Sequential([
            layers.Dropout(rate=kwargs['in_dropout_prob']),
            layers.Dense(kwargs['hidden_size'], input_dim=in_dim, activation=kwargs.get('nonlinear_fn', 'tanh')),
            layers.Dense(3, activation='softmax', use_bias=kwargs.get('bias', True))
        ])

    def call(self, text, topic):
        if self.use_topic:
            combined_input = tf.concat([text, topic], axis=1)
        else:
            combined_input = text
        y_pred = self.model(combined_input)
        return y_pred


class TGANetTF(tf.keras.Model):
    """
    Topic-grouped Attention Network
    """

    def __init__(self, **kwargs):
        super(TGANetTF, self).__init__()
        self.use_cuda = kwargs['use_cuda']
        self.hidden_dim = kwargs['hidden_size']
        self.input_dim = kwargs['text_dim']
        self.num_labels = 3
        self.attention_mode = kwargs.get('att_mode', 'text_only')
        self.learned = kwargs['learned']

        self.att = model_layers_tf.ScaledDotProductAttentionTF(self.input_dim)

        self.topic_trans = tf.Variable(tf.initializers.GlorotNormal()((kwargs['topic_dim'], self.input_dim)))

        self.ffnn = FeedForwardNeuralNetworkTF(
            use_cuda=kwargs['use_cuda'],
            add_topic=True,
            input_dim=self.input_dim,
            in_dropout_prob=kwargs['in_dropout_prob'],
            hidden_size=self.hidden_dim
        )

    def call(self, text, topic, topic_rep, text_l):
        avg_text = tf.reduce_sum(text, axis=1) / tf.expand_dims(text_l, 1)

        if topic_rep.shape[1] != topic.shape[2]:
            topic_in = tf.linalg.matmul(topic_rep, self.topic_trans)
        else:
            topic_in = topic_rep

        gen_rep = self.att.call(topic, topic_in)
        preds = self.ffnn.call(avg_text, gen_rep)

        return preds
