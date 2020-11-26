import pickle
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import preprocessing
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split, RandomizedSearchCV

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class Regressor(BaseEstimator, RegressorMixin): #inherit from these two so we can pass it as an argument into a model_selector.

    def __init__(self, x, nb_epoch = 1000, H1=18, H2=10, DROP=0, lr=0.01, batch_size=192): #now x=None so we can run HyperParamTuning
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        self.device = "cpu"
        self.x = x
        if x is not None:
            X, _ = self._preprocessor(x, training = True) 

        self.tScalerSave = None 
        self.sScalerSave = None 
        self.mmScalerSave = None
        self.labelerSave = None

        self.H1 = H1
        self.H2 = H2
        self.DROP = DROP
        self.lr = lr
        self.batch_size = batch_size

        self.input_size = X.shape[1] #columns 
        self.output_size = 1
        self.nb_epoch = nb_epoch 

        self.Network = Net(_IN=self.input_size, _H1=self.H1, _H2=self.H2, _OUT=self.output_size, _DROP=DROP)
        
        if torch.cuda.is_available():
            self.device = "cuda"
            #self.Network.cuda()
            print("CUDA GPU found")
        else:
            print("GPU was not found, defaulting to CPU") 

        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} -- Preprocessed input array of size 
                (batch_size, input_size).
            - {torch.tensor} -- Preprocessed target array of size 
                (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None

        _x = x.copy()
        if y is not None:
            _y = y.copy()
        

        # TARGET-SET INPUT DATA

        #######################################################################
        #                     -Normalize the target-                          #
        #######################################################################
        if y is not None:
          tLabel = ['median_house_value']
          tScaler = preprocessing.MinMaxScaler()
          tScaler.fit(_y.loc[:,tLabel])
          _y.loc[:,tLabel] = tScaler.transform(_y.loc[:,tLabel])

          if training:
              self.tScalerSave = tScaler #use this to compute real output using inverse_transform.
        #######################################################################

        # TRAIN-SET INPUT DATA #

        #######################################################################
        #               -Normalize the data(columns 0 to 7)-                #
        #######################################################################
        #dfLabelsStandardScaler = ['housing_median_age', 'median_income']
        dfLabelsMinMaxScaler = ['longitude', 'latitude', 'total_rooms', 'total_bedrooms', 'population', 'households', 'housing_median_age', 'median_income']
        # We can see from previous plots that h_m_a and m_i both have a sort-of normal distro. So z-norm is good there.
        # For the rest we use min-max. Z-normalization also handles outliers well (see plots)

        # Z-normalization #
        # if training:
        #     sScaler = preprocessing.StandardScaler()
        #     sScaler.fit(_x.loc[:,dfLabelsStandardScaler])
        #     self.sScalerSave = sScaler
        # else:
        #     sScaler = self.sScalerSave
        # _x.loc[:,dfLabelsStandardScaler] = sScaler.transform(_x.loc[:,dfLabelsStandardScaler])
        # Min-Max normalization #
        if training:
            mmScaler = preprocessing.MinMaxScaler()
            mmScaler.fit(_x.loc[:,dfLabelsMinMaxScaler])
            self.mmScalerSave = mmScaler
        else:
            mmScaler = self.mmScalerSave
        _x.loc[:,dfLabelsMinMaxScaler] = mmScaler.transform(_x.loc[:,dfLabelsMinMaxScaler])

        # Remember to cache the shit so we can calculate when running!
        #######################################################################

        #######################################################################
        #           -Replace all the NaN values in the DF with 0-             #
        #######################################################################
        
        _x = _x.fillna(0)
        
        # This was easy lol
        #######################################################################

        #######################################################################
        #            -Creating 1-hot encoding of ocean_proximity-             #
        #######################################################################
        oceanProximityLabels = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]

        if training:
            lb = preprocessing.LabelBinarizer()
            lb.fit(oceanProximityLabels)
            self.labelerSave = lb 
        else:
            lb = self.labelerSave

        transformedBins = lb.transform(_x.loc[:,'ocean_proximity'])
        convertedTransform = pd.DataFrame.from_records(transformedBins, columns=lb.classes_)
        _x = _x.drop(columns=['ocean_proximity'])
        _x.reset_index(drop=True, inplace=True)
        _x = pd.concat([_x, convertedTransform], axis=1, ignore_index=True)
        
        # We should now have an DateFrame with 13 columns (We removed 'ocean_proximity')
        #######################################################################

        # Once again, remember to cache the shit so we can calculate when running!
        # But only if training = True!!!!!!!!!

        ##print("proccessed")
        ##print(x)
        ##print("into")
        ##print(_x)
        
        print("_x")
        print(_x)
        if y is not None:
            print("_y")
            print(_y)

        

        return _x, (_y if isinstance(y, pd.DataFrame) else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        self.Network.to(self.device)
        optimizer = optim.Adam(self.Network.parameters(), lr=self.lr)

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget

        _X = torch.from_numpy(X.values).float().to(self.device)
        _Y = torch.from_numpy(Y.values).float().to(self.device)

        loss_function = nn.MSELoss()
        losses = []

        if next(self.Network.parameters()).is_cuda:
            print("NETWORK IS RUNNING CUDA")
        else:
            print("NETWORK IS NOTNOTNOT RUNNING CUDA")
        for epoch in range(self.nb_epoch):
            random_index = torch.randperm(_X.size(0), generator=torch.manual_seed(epoch)).to(self.device) # RNJESUS
            
            for iteration in range(_X.size(0) // self.batch_size):
                
                x_batch = _X[random_index[iteration*self.batch_size : (iteration+1)*self.batch_size]] #index the x dataset with one batch of random_index
                y_batch = _Y[random_index[iteration*self.batch_size : (iteration+1)*self.batch_size]] #index the y dataset with one batch of random_index

                optimizer.zero_grad()

                output = self.Network(x_batch)
                loss = loss_function(output, y_batch)
                loss.backward()
                
                losses.append(loss.item())
                
            if epoch%25 == 0:
                print(losses[-4:])
            optimizer.step()

        print("Losses done", losses)
            
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        
        _X = torch.from_numpy(X.values).float().to(self.device)
        

        with torch.no_grad():
            output = self.Network(_X)

        tScaler = self.tScalerSave
        trueOutput = tScaler.inverse_transform(output)
        
        return trueOutput.to_numpy()


        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).
 
        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget

        _X = torch.from_numpy(X.values).float().to(self.device)
        _Y = torch.from_numpy(Y.values).float().to(self.device)

        with torch.no_grad():
            predicted = self.Network(_X)

        loss_function = nn.MSELoss()

        loss = loss_function(predicted, _Y)

        #loss = 1 / loss

        print("Predicted:", predicted)
        print("Actual:", _Y)

        return 1 / torch.mean(loss)


        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(x_train, y_train): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    tuned_parameters = [{'x': [x_train], 
                         'H1': range(20,60), 
                         'H2': range(2, 50),
                         'batch_size': [12, 48, 96], 
                         'DROP': [0, 0.1, 0.2],
                         'lr': [0.001, 0.0005, 0.005], 
                         'nb_epoch': range(200, 800)}]


    # tuned_parameters = [{'H1': [35,40], 
    #                      'H2': [8,15], 
    #                      'batch_size': [96], 
    #                      'drop_out': [0, 0.1],
    #                      'lr': [0.0001], 
    #                      'nb_epoch': [200]}]

    search = RandomizedSearchCV(Regressor(x_train), tuned_parameters, n_iter=2, cv=3) #3-cross validation? n_iter=2? Low? no?
    search.fit(x_train, y_train)

    print("Best Score: ",round(1 / np.sqrt(search.best_score_),4),". Best parameters set found on development set:")
    print("Best Score: ",round(1 / (search.best_score_),4),". Best parameters set found on development set:")


    params = search.best_params_
    del params["x"]
    print(params)

    f = open("params.txt","w")
    f.write( str(params) )
    f.close()
    
    # means = search.cv_results_['mean_test_score']
    # stds = search.cv_results_['std_test_score']
    # for mean, std, params in zip(means, stds, search.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean, std * 2, params))

    chosen_params = (params['nb_epoch'], params['H1'], params['H2'], params['DROP'], 
                    params['lr'], params['batch_size']) #maybe more efficient method?
    return chosen_params



    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################

    #######################################################################
    #                     -Creation of Neural Net-                        #
    #######################################################################
class Net(nn.Module):

  def __init__(self, _IN, _H1, _H2, _OUT, _DROP):
       super(Net, self).__init__()

       self.linear1 = nn.Linear(_IN, _H1)
       self.linear2 = nn.Linear(_H1, _H2)
       self.linear3 = nn.Linear(_H2, _OUT)
       self.dropout = nn.Dropout(_DROP)
  
  def forward(self, x):
      
       x = F.relu(self.linear1(x))
       x = self.dropout(x)
       x = F.relu(self.linear2(x))
       x = self.dropout(x)
       x = self.linear3(x)

       return x

def anal(data):
  print("In our dataset there are {} Columns and there are {} rows".format(data.shape[0], data.shape[1]))
  print("Column names are: {}".format(data.columns))
  print("The difference types of the columns are:\n")
  print(data.dtypes)
  print("Labels for Ocean proximity:")
  print(data['ocean_proximity'].value_counts())

  fig = plt.figure()
  data.hist(figsize=(25,25),bins=50)

  

def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv") 
    #anal(data)

    # Spliting input and output
    test_split = 0.2

    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_split)

    print(x_train)
    print(y_train)

    # Training

    chosen_params = RegressorHyperParameterSearch(x_train, y_train)

    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, *chosen_params)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))

    if __name__ == "__main__":
        example_main()