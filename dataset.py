import numpy as np


def get_mmap_dataset(data_path):
    train_answers = np.load(data_path+"/train_answers.npy", mmap_mode="r")
    train_decoder_inputs = np.load(data_path+"/train_decoder_input.npy", mmap_mode="r")
    train_exercises = np.load(data_path+"/train_exercises.npy", mmap_mode="r")
    train_elapsed_times = np.load(data_path+"/train_elapsed_time.npy", mmap_mode="r")
    train_lag_times = np.load(data_path+"/train_lag_time.npy", mmap_mode="r")
    train_interval_means = np.load(data_path+"/train_interval_mean.npy", mmap_mode="r")
    train_masks = np.load(data_path+"/train_mask.npy", mmap_mode="r")


    valid_answers = np.load(data_path+"/valid_answers.npy", mmap_mode="r")
    valid_decoder_inputs = np.load(data_path+"/valid_decoder_input.npy", mmap_mode="r")
    valid_exercises = np.load(data_path+"/valid_exercises.npy", mmap_mode="r")
    valid_elapsed_times = np.load(data_path+"/valid_elapsed_time.npy", mmap_mode="r")
    valid_lag_times = np.load(data_path+"/valid_lag_time.npy", mmap_mode="r")
    valid_interval_means = np.load(data_path+"/valid_interval_mean.npy", mmap_mode="r")
    valid_masks = np.load(data_path+"/valid_mask.npy", mmap_mode="r")

    exercise_2_tag = np.load(data_path+"/exercise_2_tag.npy")
    
    train_dataset = [train_exercises, train_decoder_inputs, train_elapsed_times, train_lag_times, train_interval_means, train_answers, train_masks]
    valid_dataset = [valid_exercises, valid_decoder_inputs, valid_elapsed_times, valid_lag_times, valid_interval_means, valid_answers, valid_masks]
    
    return train_dataset, valid_dataset, exercise_2_tag



def get_batch_mmap_dataset(dataset, batch_no, batch_size):
    return [array[batch_no*batch_size:(batch_no+1)*batch_size] for array in dataset]