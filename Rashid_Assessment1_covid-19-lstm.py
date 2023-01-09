#using LSTM neural network to predict new cases (cases_new) in Malaysia using the past 30 days of number of cases.
#%%
#Import Libries
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model
from module_covid19 import EDA,ModelCreation,ModelEvaluation
import pandas as pd
import numpy as np
import datetime
import pickle
import os




#%% 
#1.0. Data path

CSV_PATH = os.path.join(os.getcwd(),'cases_malaysia_train.csv')
CSV_TEST_PATH = os.path.join(os.getcwd(),'cases_malaysia_test.csv')
MMS_PATH = os.path.join(os.getcwd(),'mms_covid_19.pkl')
#TensorBoard
log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_FOLDER_PATH = os.path.join(os.getcwd(),'logs',log_dir)

#1.2. Data loading
df = pd.read_csv(CSV_PATH,na_values='?')
df['cases_new'] = pd.to_numeric(df['cases_new'],errors='coerce') 
# to change the data type into float64 and convert blank space into NaNs

#%% 
#2.0. DATA INSPECTION
display (df.info())
stats = df.describe().T
display (df.isna().sum()) # 12 NaNs(cases_new), 342 NaNs(cluster column)
display (df.duplicated().sum()) # 0 duplicate data (of course zero haha)

eda = EDA()
eda.plot_graph(df) # to plot the graph

#%%
#3.0. DATA CLEANING 
#df['cases_new'] = df['cases_new'].interpolate()# acts like fillna for time series
df['cases_new'] = df['cases_new'].fillna(method='ffill') #fill nan for cases_new cases with the previous value.
df['cases_new'] = np.ceil(df['cases_new']) # to complete 1 body count, body count cannot be in float
display (df.isna().sum()) #verify the data has been fill clean
display (df.info())
'''
Only interpolate cases_new column, and didnt interpolating the cluster columns 
since it will not be used as features
'''

#%%
# 4.0 FEATURES SELECTION
# selecting only cases_new column.

#%%
#5.0. PREPROCESSING
# Scaling process
mms = MinMaxScaler()
df = mms.fit_transform(np.expand_dims(df['cases_new'],axis=-1))
#%%
# save using pickle
with open(MMS_PATH,'wb') as file:
    pickle.dump(mms,file)
#%%
X_train = []
y_train = []
win_size = 30

for i in range(win_size,np.shape(df)[0]):
    X_train.append(df[i-win_size:i,0])
    y_train.append(df[i,0])
    
X_train = np.array(X_train)
y_train = np.array(y_train)
# never perform train test split for time series data

#%%
# 6.0. MODEL DEVELOPMENT
mc = ModelCreation()
model = mc.simple_lstm_layer(X_train)

display (plot_model(model,show_layer_names=(True),show_shapes=(True)))

X_train = np.expand_dims(X_train,axis=-1)
#%%
# CALLBACKS
tb = TensorBoard(log_dir=LOG_FOLDER_PATH)
#%%
# Parameter
BATCH_SIZE = 32
EPOCHS = 1000

hist = model.fit(X_train,y_train,batch_size=BATCH_SIZE,epochs=EPOCHS, callbacks=(tb))

#%%
#7.0. MODEL EVALUATION
hist.history.keys()

me = ModelEvaluation()
me.plot_model_evaluation(hist)

#%%
#8.0. MODEL ANALYSIS
test_df = pd.read_csv(CSV_TEST_PATH)
test_df['cases_new'] = test_df['cases_new'].interpolate()
test_df = mms.transform(np.expand_dims(test_df['cases_new'].values,axis=-1))
con_test = np.concatenate((df,test_df),axis=0)
con_test = con_test[-130:] # 30(win_size) + 100(test_df)

X_test = []
for i in range(win_size,len(con_test)):
    X_test.append(con_test[i-win_size:i,0])
  
# Another code for line above
# X_test = [con_test[i-win_size:i,0] for i in range(win_size,len(con_test))]

X_test = np.array(X_test)

predicted = model.predict(np.expand_dims(X_test,axis=-1))

#%%
#8.1. plotting the graph
me.plot_predicted_graph(test_df, predicted, mms)

#%%
#8.2 MSE, MAPE
test_df_inversed = mms.inverse_transform(test_df)
predicted_inversed = mms.inverse_transform(predicted)

print('mae: ' + str(mean_absolute_error(test_df,predicted)))
print('mse: ' + str(mean_squared_error(test_df,predicted)))
print('mape: ' + str(mean_absolute_percentage_error(test_df,predicted)))

print('mae_i: ' + str(mean_absolute_error(test_df_inversed,predicted_inversed)))
print('mse_i: ' + str(mean_squared_error(test_df_inversed,predicted_inversed)))
print('mape_i: ' + str(mean_absolute_percentage_error(test_df_inversed,predicted_inversed)))
