# Most of the codes are taken from 'https://www.tensorflow.org/text/tutorials/transformer'
# Made the network multi-input by adding other inputs' vectors together with positional embedding
# Categorical and continuous inputs are mapped to d_model dimensional vectors with embedding and linear layer respectively.
#
# In the original code, computation order inside transformer blocks were according to the paper 'https://arxiv.org/abs/1706.03762',
# which was causing divergence issues with deep models. So, changed the order of LayerNormalization(to pre-normalization)
# Futher reading about the importance of computation order: https://arxiv.org/pdf/2002.04745.pdf

import tensorflow as tf
from utils import positional_encoding


class Linear(tf.keras.layers.Layer):
    def __init__(self, units=256, input_dim=1):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units), dtype="float32"), trainable=True)

    def call(self, inputs):
        return inputs*self.w
    
    
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])
    
    
    
def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.

    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  
    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights
    
    
    
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        x_norm = self.layernorm1(x)
        attn_output, _ = self.mha(x_norm, x_norm, x_norm, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = attn_output + x

        x_norm = self.layernorm2(out1)
        ffn_output = self.ffn(x_norm)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = out1 + ffn_output

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)


    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        x_norm = self.layernorm1(x)
        attn1, attn_weights_block1 = self.mha1(x_norm, x_norm, x_norm, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = x + attn1

        x_norm = self.layernorm2(out1)
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, x_norm, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = out1 + attn2
    
        x_norm = self.layernorm3(out2)
        ffn_output = self.ffn(x_norm)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = out2 + ffn_output

        return out3, attn_weights_block1, attn_weights_block2



class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, exercise_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        ## add exercise embedding
        self.tag_projection = tf.keras.layers.Dense(d_model)#
        #self.part_embedding = tf.keras.layers.Embedding(exercise_size, d_model)
        self.embedding = tf.keras.layers.Embedding(exercise_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                                self.d_model)


        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, tags, training, mask):

        seq_len = tf.shape(x)[1]

        tags = self.tag_projection(tags)#
        tags *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        x += tags#
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)



class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, response_size, elapsed_time_size,
               maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # add response elapsed_time and timestamp embeddings.
        self.response_embedding = tf.keras.layers.Embedding(response_size, d_model)
        self.elapsed_time_embedding = tf.keras.layers.Embedding(elapsed_time_size, d_model)
        self.lag_time_proj = Linear(d_model, 1)
        self.interval_proj = Linear(d_model, 1)

        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, response, elapsed_time, lag_time, interval_mean, enc_output, training, look_ahead_mask, padding_mask):

        seq_len = tf.shape(response)[1]
        attention_weights = {}

        response = self.response_embedding(response)  # (batch_size, target_seq_len, d_model)
        response *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        elapsed_time = self.elapsed_time_embedding(elapsed_time)  # (batch_size, target_seq_len, d_model)
        elapsed_time *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        lag_time = self.lag_time_proj(lag_time)
        lag_time *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        interval_mean = self.interval_proj(interval_mean)
        interval_mean *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        x = self.pos_encoding[:, :seq_len, :] + response + elapsed_time + lag_time + interval_mean
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
            
        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights
    
    
    
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, exercise_size, response_size, elapsed_time_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                               exercise_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                               response_size, elapsed_time_size, pe_target, rate)


        self.final_layer = tf.keras.layers.Dense(1)
        self.final_activation = tf.keras.layers.Activation("sigmoid")

    def call(self, exercises, tags, responses, elapsed_times, lag_times, interval_means, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        """
        inp: exercise
        """
        interval_means = tf.expand_dims(interval_means, axis=-1)

        enc_output = self.encoder(exercises, tags, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            responses, elapsed_times, lag_times, interval_means, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        final_output = self.final_activation(final_output)

        return final_output, attention_weights