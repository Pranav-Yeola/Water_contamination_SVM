import numpy as np
import csv
import sys
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

def import_data():
    X = np.genfromtxt("train_X_svm.csv",delimiter=',',dtype = np.float128,skip_header = 1)
    Y = np.genfromtxt("train_Y_svm.csv",delimiter=',',dtype = np.float128)
    return X, Y


def train_model(X,Y):
    clf_model = SVC(C = 0.95, kernel = 'linear',tol = 1e-4)
    clf_model.fit(X,Y)
    return clf_model


def save_model(model, model_file_name):
    with open(model_file_name, 'wb') as file:
        pickle.dump(model,file)


if __name__ == "__main__":

    X,Y = import_data()
    
    model = train_model(X,Y)
    save_model(model,"MODEL_FILE.sav")
