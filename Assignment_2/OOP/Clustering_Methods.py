from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


PLOT_PATH = "Assignment_1/Plots/"


class Clustering_Methods:

    def __init__(self, name, dataset, n_clusters):
        self.name = name
        self.dataset = dataset
        self.n_clusters = n_clusters

    def preprocess(self, df):
        """
        create x and y from a pandas dataframe
        x, which are 2D point will be scaled using min-max scaler

        :param dataframe:
        :return (Scaled X (minmax), y):
        """

        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        scaler = MinMaxScaler()
        x = scaler.fit_transform(X)

        return x, y

    def adding_methods(self):
        """
        adding all the methods with their specific names in a list

        :return: a List containing tuple of models (name of the model, model)
        """

        Models = []

        # models
        Models.append(('KMeans', self.k_means()))
        Models.append(('Gaussian Mixture', self.EM()))
        Models.append(('Spectral Clustering', self.Spectral_clustering()))

        return Models

    def Kfold_report(self, Models, x_train, y_train, dataset_name):
        """
        training all the models from the list of models using 10 fold cross validation

        :param x_train:
        :param y_train:
        :return:
        """

        print("**********")
        print("{} Dataset Results: ".format(dataset_name))

        results = []
        method_names = []
        for name, model in Models:
            # train the models
            KFold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
            CrossValidation = cross_val_score(model, x_train, y_train, cv=KFold, scoring="accuracy")
            results.append(CrossValidation)
            method_names.append(name)
            print(f"{name} Training Accuracy : {CrossValidation.mean()*100:.2f}%")

        return results, method_names


    def training_models(self, Models, x_train, x_test, y_train, y_test, dataset_name):
        """
        Training all the models from the list of models using the training set
        """
        for name, model in Models:
            model.fit(x_train)
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            self.confusion_metrics(conf_matrix, accuracy, name, dataset_name)
            self.plot_decision_boundary(model, x_test, y_test, name, dataset_name)


    def k_means(self):
        """
        create a K_means clustering model
        :return: K_means model
        """
        return KMeans(n_clusters=self.n_clusters)

    def EM(self):
        """
        create an EM clustering model
        :return: EM model
        """
        return GaussianMixture(n_components=self.n_clusters)

    def Spectral_clustering(self):
        """
        create a Spectral_clustering clustering model
        :return (name of the model, Spectral_clustering model):
        """
        return SpectralClustering(n_clusters=self.n_clusters)



    def data_spliting(self, x, y, test_size=0.2, random_state=1):
        pass



    def plotting(self, results, names, dataset_name):
        pass


    def confusion_metrics(self, conf_matrix, accuracy_score, method_name, dataset_name):
        pass


    def plot_decision_boundary(self, model, X, Y, model_name, dataset_name):
        pass








