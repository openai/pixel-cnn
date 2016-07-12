# pixel-cnn
adaptation of PixelCNN

- train_double_cnn.py trains a "2 stream" architecture consisting of 2 CNNs
- train_row_lstm.py trains a single CNN for conditioning on the pixels above the current pixel, combined with an LSTM for conditioning on the current row

both models give a test set log likelihood corresponding to 3.0-3.1 bits per dimension in < 1 day of training
