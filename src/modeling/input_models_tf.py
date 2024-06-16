import tensorflow as tf
from transformers import TFBertModel

from utilities.runtime import print_debug_message


class JointBERTLayerWithExtraTF(tf.keras.Model):
    def __init__(self, vecs, use_both=True, static_vecs=True, use_cuda=False):
        super(JointBERTLayerWithExtraTF, self).__init__()

        self.use_cuda = use_cuda
        self.static_embeds = static_vecs
        self.bert_layer = TFBertModel.from_pretrained('bert-base-uncased')

        topic_tensor = tf.convert_to_tensor(vecs)
        self.topic_embeds = tf.Variable(topic_tensor, trainable=not static_vecs)

        self.use_both = use_both

    def call(self, **kwargs):
        print_debug_message(f"{self.__name__}: call")
        text_topic = kwargs['text_topic_batch']
        token_type_ids = kwargs['token_type_ids']

        item_ids = tf.cast(text_topic, tf.int32)
        item_masks = tf.cast(tf.math.not_equal(item_ids, 0), tf.int32)
        token_type_ids = tf.cast(token_type_ids, tf.int32)

        if self.use_cuda:
            with tf.device('/GPU:0'):
                item_ids = tf.identity(item_ids)
                item_masks = tf.identity(item_masks)
                token_type_ids = tf.identity(token_type_ids)

        last_hidden, _ = self.bert_layer(item_ids, attention_mask=item_masks, token_type_ids=token_type_ids)
        full_masks = tf.expand_dims(item_masks, axis=2) * tf.ones_like(last_hidden)
        masked_last_hidden = last_hidden * tf.cast(full_masks, dtype=last_hidden.dtype)
        max_tok_len = token_type_ids.sum(1)[0].item()
        text_no_cls_sep = masked_last_hidden[:, 1:-max_tok_len - 1, :]
        topic_no_sep = masked_last_hidden[:, -max_tok_len:, :]

        text_length = tf.reduce_sum(tf.cast(tf.math.not_equal(kwargs['text'], 0), tf.int32), axis=1)
        topic_length = tf.reduce_sum(tf.cast(tf.math.not_equal(kwargs['topic'], 0), tf.int32), axis=1)

        top_v = tf.nn.embedding_lookup(
            self.topic_embeds,
            kwargs['topic_rep_ids']
        )  # Assuming 'topic_rep_ids' is indices

        avg_txt = tf.reduce_sum(text_no_cls_sep, axis=1) / tf.expand_dims(tf.cast(text_length, tf.float32), 1)
        avg_top = tf.reduce_sum(topic_no_sep, axis=1) / tf.expand_dims(tf.cast(topic_length, tf.float32), 1)

        if self.use_both:
            top_in = tf.concat([avg_top, top_v], axis=1)
        else:
            top_in = top_v

        embed_args = {
            'avg_txt_E': avg_txt,
            'avg_top_E': top_in,
            'txt_E': text_no_cls_sep,
            'top_E': topic_no_sep,
            'txt_l': text_length,
            'top_l': topic_length,
            'ori_avg_top_E': avg_top
        }

        return embed_args
