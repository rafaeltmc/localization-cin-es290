import pandas as pd
import numpy as np
#import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pickle
from PyRadioLoc.Utils.GeoUtils import GeoUtils

fingerprint_csv = pd.read_csv('fingerprint.csv')
medicoes_csv = pd.read_csv('input/testLoc.csv')

# de medicoes.csv, ver em qual fingerprint esta e calcular por como entrada o fingerprint e saida o rssi das 6 erbs
#fingerprint_csv.apply(lambda r: np.array(GeoUtils.distanceInKm(r.lat, r.lon, medicoes_csv.lat, medicoes_csv.lon)).argmin(), axis=1)


medicoes_csv['cell'] = medicoes_csv.apply(lambda r: np.array(GeoUtils.distanceInKm(r.lat, r.lon, fingerprint_csv.lat, fingerprint_csv.lon)).argmin(), axis=1)
medicoes_csv.to_csv('testLoc_cells.csv',index=False)

"""
X = medicoes_csv['cell']
y = medicoes_csv.drop('cell',axis=1).drop('lat',axis=1).drop('lon',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

lm = LinearRegression()
lm.fit(X_train, y_train)

y_pred = lm.predict(X_test)

plt.scatter(y_pred,y_test)
plt.xlabel("Predito")
plt.ylabel("Real")
"""