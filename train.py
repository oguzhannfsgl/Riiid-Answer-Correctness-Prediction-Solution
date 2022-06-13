import tensorflow as tf
import yaml
from tqdm import tqdm

from utils import *
from dataset import *
from model import Transformer



@tf.function
def train_step(model, optimizer, masked_loss, auc_object, loss_object, input_batch):
    exercises_batch, decoder_inputs_batch = input_batch[0], input_batch[2]
    y_real, mask = input_batch[-2], input_batch[-1]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(exercises_batch, decoder_inputs_batch)

    with tf.GradientTape() as tape:
        preds, _ = model(*input_batch[:-2], True, enc_padding_mask, combined_mask, dec_padding_mask)
        loss = masked_loss(y_real, preds, mask)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    masked_y_real = tf.boolean_mask(y_real, mask)
    masked_y_pred = tf.boolean_mask(preds, mask)
    
    auc_object.update_state(tf.keras.backend.flatten(masked_y_real), tf.keras.backend.flatten(masked_y_pred))
    loss_object(loss) # Just to pass loss information to the metrics.Mean object to print later
    
    
@tf.function
def valid_step(model, auc_object, loss_object, input_batch):
    exercises_batch, decoder_inputs_batch = input_batch[0], input_batch[2]
    valid_answers_batch = input_batch[-2]
    y_real, mask = input_batch[-2], input_batch[-1]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(exercises_batch, decoder_inputs_batch)

    preds, _ = model(*input_batch[:-2], False, enc_padding_mask, combined_mask, dec_padding_mask)
    loss = masked_loss(y_real, preds, mask)

    masked_y_real = tf.boolean_mask(y_real, mask)
    masked_y_pred = tf.boolean_mask(preds, mask)
    
    auc_object.update_state(tf.keras.backend.flatten(masked_y_real), tf.keras.backend.flatten(masked_y_pred))        
    loss_object(loss)
                  
                  
with open("config.yaml") as f:
    config = yaml.safe_load(f)

epochs, batch_size = config["epochs"], config["batch_size"]


masked_loss = get_masked_bce_loss()
loss = tf.keras.metrics.Mean(name='train_loss')
auc_score = tf.keras.metrics.AUC()
valid_auc_score = tf.keras.metrics.Mean()

train_iter_size = (config["train_samples"]//config["batch_size"])+1
valid_iter_size = (config["valid_samples"]//config["batch_size"])+1

model = Transformer(config["num_layers"], config["d_model"], config["num_heads"], config["dff"], config["exercise_size"], config["response_size"], config["elapsed_time_size"], config["seq_len"], config["seq_len"], config["dropout"])

lrs = get_lrs(config["train_samples"], config["batch_size"], config["epochs"], config["warmup_steps"], config["d_model"])
optimizer = tf.keras.optimizers.Adam(learning_rate=lrs[0], beta_1=0.9, beta_2=0.98, epsilon=1e-9)
optimizer.learning_rate.assign(tf.Variable(lrs[0], dtype="float64"))
last_step = 1


train_dataset, valid_dataset, exercise_2_tag = get_mmap_dataset(config["data_path"])

step = 0
for epoch in range(epochs):
    auc_score.reset_states()
    pbar = tqdm(range(train_iter_size))
    for batch_no in pbar:
        train_batch = get_batch_mmap_dataset(train_dataset, batch_no, batch_size)
        train_tags_batch = exercise_2_tag[train_batch[0].astype("int16")]
        train_batch.insert(1, train_tags_batch)
        
        loss.reset_states()
        train_step(model, optimizer, masked_loss, auc_score, loss, train_batch)
        optimizer.learning_rate.assign(tf.Variable(lrs[step], dtype="float64"))
        step += 1
        pbar.set_description(f"Epoch:{epoch+1} - Training AUC:{auc_score.result()} - Training Loss:{loss.result()}")
    

    pbar = tqdm(range(valid_iter_size))
    valid_aucs = []
    auc_score.reset_states()
    for batch_no in pbar:
        valid_batch = get_batch_mmap_dataset(valid_dataset, batch_no, batch_size)
        valid_tags_batch = exercise_2_tag[valid_batch[0].astype("int16")]
        valid_batch.insert(1, valid_tags_batch)
            
        loss.reset_states()

        valid_step(model, auc_score, loss, valid_batch)
        pbar.set_description(f"Validation AUC:{auc_score.result()} - Validation Loss:{loss.result()}")
        
model.save_weights(f"transformer_riiid_weights")