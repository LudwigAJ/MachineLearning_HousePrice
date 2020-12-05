## Predicting californian house prices based on data from US census bureau in the 90's

## Instructions
### In terminal, run "python3 /PATH/TO/FILE.py" without quotation marks.

### File can be modified within the function 'example_main()' to include more parameters to do Cross-Validation on.
### The current optimizer (object that updates the weights after backpropogation) uses ADAM, but it could easily be changed into SGD in one line of code inside 
### 'fit()' function

## Design

#### The model implements a 3 layer Neural Network implementing reLU and drop out
#### It uses libraries: pytorch(for backprop and forward), sklearn(for pre-processing and hyper-parameter tuning), numpy(for other matrix calculations e.t.c),
#### pandas(for reading csv and data processing), matplotlib(for analyzing data visually), and pickle(for 'pickling' objects).
#### The model tries different parameters using Randomized Search Cross Validation. It currently uses ADAM to update weights of Neural Network.


