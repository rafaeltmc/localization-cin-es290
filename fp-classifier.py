import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import MinMaxScaler
from PyRadioLoc.Utils.GeoUtils import GeoUtils



from matplotlib import pyplot as plt

MIN_LAT = -8.080
MIN_LON = -34.91
MAX_LAT = -8.065
MAX_LON = -34.887
rssi_fitter = pd.DataFrame(data={'RSSI_1': [-150, 0], 'RSSI_2': [-150, 0], 'RSSI_3': [-150, 0], 'RSSI_4': [-150, 0], 'RSSI_5': [-150, 0], 'RSSI_6': [-150, 0]})
                                    #,'lat1': [MIN_LAT, MAX_LAT], 'lon1': [MIN_LAT, MAX_LAT],
                                    #'lat2': [MIN_LAT, MAX_LAT], 'lon2': [MIN_LAT, MAX_LAT],
                                    #'lat3': [MIN_LAT, MAX_LAT], 'lon3': [MIN_LAT, MAX_LAT],
                                    #'lat4': [MIN_LAT, MAX_LAT], 'lon4': [MIN_LAT, MAX_LAT],
                                    #'lat5': [MIN_LAT, MAX_LAT], 'lon5': [MIN_LAT, MAX_LAT],
                                    #'lat6': [MIN_LAT, MAX_LAT], 'lon6': [MIN_LAT, MAX_LAT]
                                    #,'erb_dist_0_1': [0, 2], 'erb_dist_0_2': [0, 2], 'erb_dist_0_3': [0, 2], 'erb_dist_0_4': [0, 2], 'erb_dist_0_5': [0, 2], 'erb_dist_1_2': [0, 2], 'erb_dist_1_3': [0, 2], 'erb_dist_1_4': [0, 2], 'erb_dist_1_5': [0, 2], 'erb_dist_2_3': [0, 2], 'erb_dist_2_4': [0, 2], 'erb_dist_2_5': [0, 2], 'erb_dist_3_4': [0, 2], 'erb_dist_3_5': [0, 2], 'erb_dist_4_5': [0, 2]})
lat_lon_fitter = pd.DataFrame(data={'lat': [MIN_LAT, MAX_LAT], 'lon': [MIN_LON, MAX_LON]})

scaler1 = MinMaxScaler()
scaler1.fit(rssi_fitter)

scaler2 = MinMaxScaler()
scaler2.fit(lat_lon_fitter)

fingerprint_csv = pd.read_csv('fingerprint.csv')
medicoes_csv = pd.read_csv('medicoes_cell.csv')
#medicoes_csv = pd.read_csv('input/medicoes.csv')
erbs_csv = pd.read_csv('input/erbs.csv')
#test_csv = pd.read_csv('input/testLoc.csv')
test_csv = pd.read_csv('testLoc_cells.csv')
'''
for i in range(1,7):
    medicoes_csv['lat'+str(i)] = erbs_csv.lat[i-1]
    medicoes_csv['lon'+str(i)] = erbs_csv.lon[i-1]
    test_csv['lat'+str(i)] = erbs_csv.lat[i-1]
    test_csv['lon'+str(i)] = erbs_csv.lon[i-1]



for i in range(0,6):
    for j in range(i+1,6):
        medicoes_csv['erb_dist_'+str(i)+'_'+str(j)] = medicoes_csv.apply(lambda x: GeoUtils.distanceInKm(erbs_csv.lat[i], erbs_csv.lon[i], erbs_csv.lat[j], erbs_csv.lon[j]), axis=1)
        test_csv['erb_dist_'+str(i)+'_'+str(j)] = test_csv.apply(lambda x: GeoUtils.distanceInKm(erbs_csv.lat[i], erbs_csv.lon[i], erbs_csv.lat[j], erbs_csv.lon[j]), axis=1)
'''

X = medicoes_csv.iloc[:,[2,3,4,5,6,7]].values
#y = medicoes_csv.iloc[:,[0,1]].values
#y = fingerprint_csv.iloc[  medicoes_csv.iloc[:,8].values.tolist()  ,[0,1]]
y = medicoes_csv.iloc[:,[8]].values.ravel()
X_val =	test_csv.iloc[:,[2,3,4,5,6,7]].values
y_val = test_csv.iloc[:,0:2].values
#y_val = fingerprint_csv.iloc[  test_csv.iloc[:,8].values.tolist()  ,[0,1]].values



X = scaler1.transform(X)
X_val = scaler1.transform(X_val)
#y = scaler2.transform(y)

neigh = KNeighborsClassifier()
neigh.fit(X, y)
#y_pred = scaler2.inverse_transform(neigh.predict(X_val))
y_pred = neigh.predict(X_val)

df = pd.DataFrame({})


df['p_lat'] = fingerprint_csv.iloc[ y_pred  ,[0]].values.ravel()
df['p_lon'] = fingerprint_csv.iloc[ y_pred  ,[1]].values.ravel()
df['o_lat'] = y_val[:,0]
df['o_lon'] = y_val[:,1]

df['dist'] = df.apply(lambda x: GeoUtils.distanceInKm(x.p_lat, x.p_lon, x.o_lat, x.o_lon), axis=1)
 
erro = np.mean(df['dist'].values)

print("Max: ", df['dist'].values.max())
print("Min: ", df['dist'].values.min())
print("Erro: ", erro)


plt.scatter([0]*df.shape[0],df['dist'],s=1)
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_visible(False)
plt.show()