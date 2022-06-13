import numpy as np
import tensorflow as tf


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)




def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by 
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask

def get_masked_bce_loss():
    bce = tf.keras.losses.BinaryCrossentropy()
    def masked_loss(y_real, y_pred, mask):
        masked_y_real = tf.boolean_mask(y_real, mask)
        masked_y_pred = tf.boolean_mask(y_pred, mask)
        return bce(masked_y_real,masked_y_pred)
    return masked_loss


def lr_schedule(step, max_step, warmup_steps, d_model, last_lr):
    if step<max_step:
        return (d_model**-0.5)*min([step**-0.5, step*warmup_steps**-1.5])
    else:
        return last_lr * (0.95 ** np.sqrt(step - max_step))

def get_lrs(train_samples, batch_size, epochs, warmup_steps, d_model):
    max_step = ((train_samples/batch_size) + 1) * (epochs-1)
    steps = np.arange(1, epochs*max_step/(epochs-1))
    last_lr = (d_model**-0.5)*min([max_step**-0.5, max_step*warmup_steps**-1.5])
    lrs = [lr_schedule(step, max_step, warmup_steps, d_model, last_lr) for step in steps]
    return lrs
