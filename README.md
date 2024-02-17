# CNNforStellarp
Deep learning predictions of galaxy stellar populations in the low-redshift Universe

We give the trained model and a SDSS spectrum. One can load the trained model to predict the four parameters: VD, logAge, MH and E(B-V)  of a spectrum.
Note that the four predicted values from the model are scaled to a given range using sklearn.preprocessing.MinMaxScaler, so the predicted values must be inverse transformed to the original scale.

Note that the model file is larger than 1GB, if one needs this file, please send an email to the author( jsjxwll@126.com) .

