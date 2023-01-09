import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout
from tensorflow.keras import Input
import numpy as np

class EDA():
    def __init__(self):
        pass
    
    def plot_graph(self,df):
        '''
        This function is to plot the graph

        '''
        plt.figure()
        plt.plot(df['cases_new'])
        plt.plot(df['cases_active'])
        plt.plot(df['cases_recovered'])
        plt.legend(['cases_new','cases_active','cases_recovered'])
        plt.title('Malaysia Covid-19 Cases')
        plt.show()

class ModelCreation():
    def __init__(self):
        pass
    
    def simple_lstm_layer(self,X_train,num_node=16,drop_rate=0.03,activation='relu',output_node=1): # default activation is relu
        model = Sequential()
        model.add(Input(shape=(np.shape(X_train)[1],1))) # input length # features
        model.add(LSTM(num_node)) # LSTM
        model.add(Dropout(drop_rate))
        model.add(Dense(output_node,activation)) # output layer
        model.compile(optimizer='adam',loss='mse',metrics='mape')
        model.summary()
        
        return model

class ModelEvaluation():
    def __init__(self):
        pass
    
    def plot_model_evaluation(self,hist):
        hist_keys = [i for i in hist.history.keys()]
        plt.figure()
        plt.plot(hist.history[hist_keys[0]])
        plt.title('Loss')
        plt.show()

        plt.figure()
        plt.plot(hist.history[hist_keys[1]])
        plt.title('MAPE')
        plt.show()
        
    def plot_predicted_graph(self,test_df,predicted,mms):
        plt.figure()
        plt.plot(test_df,'b',label='actual_new_cases')
        plt.plot(predicted,'r',label='predicted_new_cases')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(mms.inverse_transform(test_df),'b',label='actual_new_cases')
        plt.plot(mms.inverse_transform(predicted),'r',label='predicted_new_cases')
        plt.legend()
        plt.title('Malaysia Covid-19 Cases Trendline')
        plt.show()
        