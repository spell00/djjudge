
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
