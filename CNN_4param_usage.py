import os
import joblib
import pandas as pd
from astropy.io import fits
import numpy as np
import tensorflow as tf
from keras import backend as K

def interpolate_arr(wave_new, wave_original, flux_original):
    """
    Interpolate the flux values for a new set of wavelengths.

    Args:
        wave_new (np.array): The new wavelength array.
        wave_original (np.array): The original wavelength array.
        flux_original (np.array): The original flux array.

    Returns:
        flux_new(np.array): The interpolated flux array.
    """
    flux_new = np.interp(wave_new, wave_original, flux_original)
    return flux_new


def normalize_5500(wave, flux):
    """
    Normalize the flux by flux around the 5500A

    Args:
        wave (np.array): The wavelength array.
        flux (np.array): The flux array.

    Returns:
        flux(np.array): The normalized flux array.
    """
    n = np.where((wave > 5500 - 1) & (wave < 5500 + 1))
    flux5500 = np.median(flux[n])
    flux = flux / flux5500
    return flux
def getfits(file_dir,fitsname):
    """
    Process a list of FITS files to extract and normalize flux values, interpolate to fixed dimensions,
    and return the interpolated flux values.

    Args:
        file_dir (str): The directory where the FITS files are located.
        fitsname (list): A list of FITS file names to process.

    Returns:
        flux_interp_save(list): A list containing the interpolated flux values for each FITS file.
    """
    # Interpolate to ensure that all flux arrays have the same dimensions.
    flux_interp_save=[]# Store the interpolated flux
    wave_int=np.arange(3800,8900,1)# Construct the interpolated wavelength
    for specname in fitsname:
        hdulist = fits.open(file_dir + specname)
        t = hdulist[1].data
        original_flux = t['flux']
        wave = t['loglam']
        wave = 10 ** wave
        flux_norm = normalize_5500(wave, original_flux)  # Normalize the flux        
        flux_interpolate = interpolate_arr(wave_int, wave, flux_norm)  # # Interpolate to fixed dimensions
        flux_interp_save.append(flux_interpolate)  # Add the interpolated flux to the list
    return flux_interp_save

def modelpredict(X):
    """
    load the trained model to predict the four parameters: VD, logAge, MH and E(B-V)  of a spectrum.
    Note that the four predicted values from the model are scaled to a given range
    using sklearn.preprocessing.MinMaxScaler, so the predicted values must be inverse transformed to the original scale.

    Args:
        X (numpy.ndarray): Input data in the form of a numpy array.

    Returns:
       y_pred_(numpy.ndarray): A numpy array containing the predicted target values.
    """
    model = tf.keras.models.load_model("model_4para.h5", custom_objects={'coeff_determination': coeff_determination})
    X = np.array(X).reshape(-1, X.shape[1], 1).astype("float32")
    y_pred_ = model.predict(X)
    y_pred_ =  np.array(y_pred_).reshape(-1, y_pred_.shape[1], 1).astype("float32")
    print(y_pred_.shape[1])
    for i in range(4):
        min_max_scaler = joblib.load('./min_max_scaler/scalar0'+str(i+1))
        y_pred_[:, i] = min_max_scaler.inverse_transform(y_pred_[:,i])
    return y_pred_


def coeff_determination(y_true, y_pred):
    """
       The coefficient of determination R^2 is often used in linear regression to represent the percentage of the dependent variable's variance explained by the regression line. If R^2 = 1, it indicates that the model perfectly predicts the target variable.
       Formula: R^2 = SSR/SST = 1 - SSE/SST
       Where: SST (total sum of squares) is the total sum of squares, SSR (regression sum of squares) is the regression sum of squares, and SSE (error sum of squares) is the residual sum of squares.

       Args:
           y_true (array-like): True values of the dependent variable.
           y_pred (array-like): Predicted values of the dependent variable.

       Returns:
           float: Coefficient of determination (R^2).

       """
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))
def getdataname(path):
    """
    Read the names of all files in a folder and store them in a list.

    Args:
        path (str): The path to the folder containing the files.

    Returns:
        list: A list containing the names of all files in the folder.
    """
    datanames = os.listdir(path)
    list = []
    for i in datanames:
        list.append(i)
    return list

path = './data/'
specname=getdataname(path)
X=np.array(getfits(path,specname))
y_pred_=modelpredict(X)
## Save as CSV
df_c=pd.DataFrame()
df_c['fitsname']=np.array(specname)
df_c['VD']=y_pred_[:,0]
df_c['LogAge']=y_pred_[:,1]
df_c['[M/H]']=y_pred_[:,2]
df_c['E(B-V)']=y_pred_[:,3]
df_c.to_csv("result.csv",index=False)