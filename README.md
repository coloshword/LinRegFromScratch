### 10/20/24:

1) Linear regression module from scratch

2) Simple application of the module 


### 11/17/24:
Fixed an error with my values exploding, because we needed to use scaling / normalization of the input feature values. With that, our values become much closer. In addition, increasing the number of epochs helps alot. My implementation uses a default of 5 epochs, which for this small dataset is not enough. Increasing epochs to 100 increased my model performance by oom. 

After some tuning, I see that for my housing dataset, training it with 1,000 epochs and a learning rate of 0.005, I get performance basically equivalent to scikit learn's LinearRegression module. Seems to be that since the dataset is small (about 450 training examples), more passes allowed the model to converge better. 