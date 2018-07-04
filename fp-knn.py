import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle

fingerprint_csv = pd.read_csv('fingerprint.csv')
cell_rssi_csv = pd.read_csv('cell_rssi.csv')


rssi_fitter = pd.DataFrame(data={'RSSI_1': [-150, 0], 'RSSI_2': [-150, 0], 'RSSI_3': [-150, 0], 'RSSI_4': [-150, 0], 'RSSI_5': [-150, 0], 'RSSI_6': [-150, 0]})
scaler1 = MinMaxScaler()
scaler1.fit(rssi_fitter)


features = ['RSSI_1','RSSI_2','RSSI_3','RSSI_4','RSSI_5','RSSI_6']

data = cell_rssi_csv[features]

#target = cell_rssi_csv.iloc[:,0]
target = cell_rssi_csv.iloc[:,0].values.tolist()
target = fingerprint_csv.iloc[target,[0,1]]

#data = scaler1.transform(data)

knn = KNeighborsRegressor()

knn.fit(data, target)

with open('knn2.pkl', 'wb') as output:
    pickle.dump(knn, output, pickle.HIGHEST_PROTOCOL)