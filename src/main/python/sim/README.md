
## Getting Started

install Nvidia's Apex package

pip install -r requirements.txt

### Train the convolutional Variational Autoencoder (VAE) 
(Before doing this, the training_folders must be changed with paths of folders containing .wav files only)

python train_ae.py

### Train the 1D-CNN (to be verified, might need small modifications)
(Before doing this, the training_folders must be changed with paths of folders containing .wav files only and scores should point to the scores files, in the same order)

python train.py


** Note: a GPU is needed. removing all .cuda() calls would be required to work on CPU. It might be hopeless to train this on a CPU, but inference should be possible if pretrained models are available.


Problems with learning the regression: the model first learns the average score of the samples, which is 0.4 in my case. It gets a significant decrease when it happens. Then, the slope goes up slowly, 
but it is not satisfying. The score tends to stay quite conservative, around the average. It does not help that there is a higher concentration of scores around the average and the scores with the lowest 
probabilities in the dataset are the highest scores. My hypothesis is that the scores close to the mean are rapidely the most accurate score and because they are also the most frequent, it might cost a lot
more to find a good configuration for the extreme scores. This is unfortunate, because the highest scores are the most interesting. I don't think it is of any interest to use a classifier only good at 
predicting bad or average material. In contrast, a classifier that would be very good at recognizing good material but bad at distinguishing between bad and average would be much more interesting. 