import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from PyRadioLoc.Utils.GeoUtils import GeoUtils
from matplotlib import pyplot as plt
import pickle

rssi_fitter = pd.DataFrame(data={'RSSI_1': [-150, 0], 'RSSI_2': [-150, 0], 'RSSI_3': [-150, 0], 'RSSI_4': [-150, 0], 'RSSI_5': [-150, 0], 'RSSI_6': [-150, 0]})
scaler1 = MinMaxScaler()
scaler1.fit(rssi_fitter)

with open('knn2.pkl', 'rb') as infile:
    knn = pickle.load(infile)

    fingerprint_csv = pd.read_csv('fingerprint.csv')
    medicoes_csv = pd.read_csv('input/testLoc.csv')
    
    features = ['RSSI_1','RSSI_2','RSSI_3','RSSI_4','RSSI_5','RSSI_6']

    data = medicoes_csv[features]

    data = scaler1.transform(data)

    y_pred = knn.predict(data)

    #cells = np.rint(y_pred).astype(int).tolist()

    #medicoes_csv['p_lat'] = fingerprint_csv.iloc[cells].lat.values.tolist()
    #medicoes_csv['p_lon'] = fingerprint_csv.iloc[cells].lon.values.tolist()

    medicoes_csv['p_lat'] = y_pred[:,0]
    medicoes_csv['p_lon'] = y_pred[:,1]
    medicoes_csv['dist'] = medicoes_csv.apply(lambda row: GeoUtils.distanceInKm(row.lat, row.lon, row.p_lat, row.p_lon),  axis=1)

    erro = np.mean(medicoes_csv['dist'].values)

    print("Max: ", medicoes_csv['dist'].values.max())
    print("Min: ", medicoes_csv['dist'].values.min())
    print("Erro: ", erro)


    plt.scatter([0]*medicoes_csv.shape[0],medicoes_csv['dist'],s=1)
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_visible(False)
    plt.show()