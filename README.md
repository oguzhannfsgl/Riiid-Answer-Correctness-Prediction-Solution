# Riiid-Answer-Correctness-Prediction-Solution
Top 5%(163/3395) solution of the Kaggle challenge 'Riiid Answer Correctness Prediction'
 
### Competition
Aim of the competition is to predict whether the student will answer the next question correctly or not. Dataset is provided as a pandas dataframe. 
Competition link is [here](https://www.kaggle.com/competitions/riiid-test-answer-prediction).

### Work
I did some feature engineering and converted the data into .npy file. Link to the this processed data is provided in the notebook. Sequence is padded/trimmed to 256 for each student. Dataset used in this work is explained below, but you can get more information from the competition link.

#### Description of the data
answers: A binary value indicates whether the student correctly answered the question or not.

decoder_input: Left shifted version of the answers that will feed the decoder part of the Transformer.

exercises: ID code for the user interaction.

elapsed_times: The average time in milliseconds it took a user to answer each question in the previous question bundle, ignoring any lectures in between. This feature is normalized with minimum and maximum values of it.

lag_time: The time between the current and last question. Since the values are too far from each other(sparse) and there are clusters in different ranges, this feature is log-normalized.

mask: A binary mask, which is 0 for paddings.

exercises_2_tag: A dictionary maps exercises into tags.

tag: one tag codes for the lecture. The meaning of the tags will not be provided, but these codes are sufficient for clustering the lectures together.


#### Notes
Transformer model was way better than LGBM. Transformer is used in the winning solution of the competition, too. Apart from main features that is directly provided from the host, log-normalized lag_time feature had the most impact on the result(Directly normalized lag_time degregaded the performance).

I spent most of the time working on features and make transformer work with deep encoder and decoder blocks. I solved the divergence issue with deep architectures in the last week and could make only two training session.
