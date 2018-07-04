import pandas as pd
import numpy as np
import math
from osgeo import ogr
from osgeo import osr

import matplotlib.pyplot as plt
from PyRadioLoc.Pathloss.Models import FreeSpaceModel
from PyRadioLoc.Pathloss.Models import FlatEarthModel
from PyRadioLoc.Pathloss.Models import LeeModel
from PyRadioLoc.Pathloss.Models import EricssonModel
from PyRadioLoc.Pathloss.Models import Cost231Model
from PyRadioLoc.Pathloss.Models import Cost231HataModel
from PyRadioLoc.Pathloss.Models import OkumuraHataModel
from PyRadioLoc.Pathloss.Models import Ecc33Model
from PyRadioLoc.Pathloss.Models import SuiModel
from PyRadioLoc.Utils.GeoUtils import GeoUtils

def fingerprint(size):
    MIN_LAT = -8.080
    MIN_LON = -34.91
    MAX_LAT = -8.065
    MAX_LON = -34.887

    # Create working area (square)
    square = {
        'topLeft': ogr.Geometry(ogr.wkbPoint),
        'bottomLeft': ogr.Geometry(ogr.wkbPoint),
        'topRight': ogr.Geometry(ogr.wkbPoint),
        'bottomRight': ogr.Geometry(ogr.wkbPoint)
    }
    square['topLeft'].AddPoint_2D(MIN_LAT, MAX_LON)
    square['bottomLeft'].AddPoint_2D(MIN_LAT, MIN_LON)
    square['topRight'].AddPoint_2D(MAX_LAT, MAX_LON)
    square['bottomRight'].AddPoint_2D(MAX_LAT, MIN_LON)

    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(4326)
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(32725) # 32725: WGS 84 / UTM zone 25S / 31985: SIRGAS 2000 / UTM zone 25S
    coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    
    square['topLeft'].Transform(coordTransform)
    square['bottomLeft'].Transform(coordTransform)
    square['topRight'].Transform(coordTransform)
    square['bottomRight'].Transform(coordTransform)

    # Calculate distance between minimum and maximum latitudes
    latDistance = GeoUtils.distanceInKm(MIN_LAT, MIN_LON, MAX_LAT, MIN_LON)*1000 #square['topLeft'].Distance(square['topRight'])
    # Get latitude increment
    latIncrement =  math.fabs(MIN_LAT - MAX_LAT) / (latDistance/size)
    # Get latitude offset
    latOffset = latIncrement * math.modf(latDistance/size)[0] / 2

    # Calculate distance between minimum and maximum longitudes
    lonDistance = GeoUtils.distanceInKm(MIN_LAT, MIN_LON, MIN_LAT, MAX_LON)*1000 #square['topLeft'].Distance(square['bottomLeft'])
    # Get longitude increment
    lonIncrement = math.fabs(MIN_LON - MAX_LON) / (lonDistance/size)
    # Get longitude offset
    lonOffset = lonIncrement * math.modf(lonDistance/size)[0] / 2

    # Calculate range of coordinates
    # Create two lat ranges, one for evens rows and another for odds
    # The odds rows must be shifted by latIncrement/2
    # So, the topography is changed from: *  *  * to: * * *
    #                                     *  *  *      * *
    #                                     *  *  *     * * *
    # Create also two lon ranges that complements each other 

    latRangeEvens = np.arange(MIN_LAT + latOffset, MAX_LAT, latIncrement)
    lonRangeEvens = np.arange(MIN_LON + lonOffset, MAX_LON, 2 * lonIncrement)

    latRangeOdds = np.arange(MIN_LAT + latOffset + latIncrement/2, MAX_LAT, latIncrement) 
    lonRangeOdds = np.arange(MIN_LON + lonOffset + lonIncrement, MAX_LON, 2 * lonIncrement)

    # Calculate each point by doing the cartesian product of the ranges
    fpCoordsEvens = np.transpose([np.tile(latRangeEvens, len(lonRangeEvens)), np.repeat(lonRangeEvens, len(latRangeEvens))])
    fpCoordsOdds = np.transpose([np.tile(latRangeOdds, len(lonRangeOdds)), np.repeat(lonRangeOdds, len(latRangeOdds))])

    # Concatenate both coords list
    fpCoords = np.concatenate((fpCoordsEvens, fpCoordsOdds), axis=0)

    plt.scatter(fpCoords[:,0], fpCoords[:,1])
    plt.show()

    return pd.DataFrame(data=fpCoords, columns=['lat','lon'])

def fingerprint_erbs_pathloss(fpCoords, erbs_csv):
    okumura = OkumuraHataModel(900)
    fpCoords['RSSI_1'] = okumura.pathloss(GeoUtils.distanceInKm(erbs_csv.lat[0], erbs_csv.lon[0], fpCoords.lat, fpCoords.lon))
    fpCoords['RSSI_2'] = okumura.pathloss(GeoUtils.distanceInKm(erbs_csv.lat[1], erbs_csv.lon[1], fpCoords.lat, fpCoords.lon))
    fpCoords['RSSI_3'] = okumura.pathloss(GeoUtils.distanceInKm(erbs_csv.lat[2], erbs_csv.lon[2], fpCoords.lat, fpCoords.lon))
    fpCoords['RSSI_4'] = okumura.pathloss(GeoUtils.distanceInKm(erbs_csv.lat[3], erbs_csv.lon[3], fpCoords.lat, fpCoords.lon))
    fpCoords['RSSI_5'] = okumura.pathloss(GeoUtils.distanceInKm(erbs_csv.lat[4], erbs_csv.lon[4], fpCoords.lat, fpCoords.lon))
    fpCoords['RSSI_6'] = okumura.pathloss(GeoUtils.distanceInKm(erbs_csv.lat[5], erbs_csv.lon[5], fpCoords.lat, fpCoords.lon))



erbs_csv = pd.read_csv('input/erbs.csv')

fpCoords = fingerprint(20)
#fingerprint_erbs_pathloss(fpCoords, erbs_csv)

fpCoords.to_csv('fingerprint.csv',index=False)