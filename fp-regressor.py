import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pickle
from PyRadioLoc.Utils.GeoUtils import GeoUtils

fingerprint_csv = pd.read_csv('fingerprint.csv')
medicoes_csv = pd.read_csv('medicoes_cell.csv')

rssi_fitter = pd.DataFrame(data={'RSSI_1': [-150, 0], 'RSSI_2': [-150, 0], 'RSSI_3': [-150, 0], 'RSSI_4': [-150, 0], 'RSSI_5': [-150, 0], 'RSSI_6': [-150, 0]})
scaler1 = MinMaxScaler()
scaler1.fit(rssi_fitter)


outputs = ['RSSI_1','RSSI_2','RSSI_3','RSSI_4','RSSI_5','RSSI_6']
X = medicoes_csv.iloc[:,8].values.tolist()
X = fingerprint_csv.iloc[X,[0,1]]
y = medicoes_csv[outputs]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=100)

y_train = scaler1.transform(y_train)

lm = KNeighborsRegressor()
lm.fit(X_train, y_train)

#cells = list(range(0, fingerprint_csv.shape[0]))

#rssi_cells = lm.predict(np.array(cells).reshape(-1, 1))

rssi_cells = lm.predict(fingerprint_csv.iloc[:,[0,1]])

outputDf = pd.DataFrame(rssi_cells)
outputDf.columns = outputs

outputDf.to_csv('cell_rssi.csv')

'''
y_pred = lm.predict(X_test)

plt.scatter(y_pred[:,1],y_test['RSSI_2'])
plt.xlabel("Predito")
plt.ylabel("Real")
sns.distplot(y_pred[:,1]-y_test['RSSI_2'])

plt.show()
'''