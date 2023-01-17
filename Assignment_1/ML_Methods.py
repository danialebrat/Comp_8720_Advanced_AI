from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, plot_confusion_matrix
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, RepeatedKFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn_evaluation.plot import grid_search


PLOT_PATH = "Assignment_1/Plots/"


class ML_Methods:

    def __init__(self, name, dataset):
        self.name = name
        self.dataset = dataset

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
        Models.append(self.QDA())
        Models.append(self.LDA())
        Models.append(self.KNN())
        Models.append(self.Gaussian_Naive_Bayes())
        Models.append(self.Bernoulli_Naive_Bayes())
        Models.append(self.Multinomial_Naive_Bayes())

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


    def training_models(self, Models, x_train, x_test, y_train, y_test, datasetname):

        for name, model in Models:
            model.fit(x_train, y_train)
            predicted = model.predict(x_test)
            cm = confusion_matrix(y_test, predicted)
            AS = accuracy_score(y_test, predicted)

            self.confusion_metrics(cm, AS, name, datasetname)

    def QDA(self):
        """
        create a quadratic-discriminant-analysis classifier
        :return (name of the mode, QDA model):
        """
        name = "QDA"
        QDA_model = QuadraticDiscriminantAnalysis()
        return (name, QDA_model)

    def LDA(self):
        """
        create a linear-discriminant-analysis classifier
        :return (name of the mode, QDA model):
        """
        name = "LDA"
        clf = LinearDiscriminantAnalysis()
        return (name, clf)

    def KNN(self):
        """
        create a KNN classifier
        :return (name of the mode, KNN model):
        """
        name = "KNN"
        KNN_Model = KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=2)
        return (name, KNN_Model)

    def Gaussian_Naive_Bayes(self):
        """
        create a Gaussian Naive Bayes classifier
        :return (name of the mode, Naive Bayes model):
        """
        name = "GNB"
        Gaussian_Naive_Model = GaussianNB()
        return (name, Gaussian_Naive_Model)

    def Bernoulli_Naive_Bayes(self):
        """
        create a Bernoulli Naive Bayes classifier
        :return (name of the mode, Naive Bayes model):
        """
        name = "BNB"
        Bernoulli_Naive_Model = BernoulliNB()
        return (name, Bernoulli_Naive_Model)

    def Multinomial_Naive_Bayes(self):
        """
        create a Multinomial Naive Bayes classifier
        :return (name of the mode, Naive Bayes model):
        """
        name = "MNB"
        Multinomial_Naive_Model = MultinomialNB()
        return (name, Multinomial_Naive_Model)

    def SVM_Linear(self):
        """
        create a SVM classifier
        :return (name of the mode, QDA model):
        """
        name = "SVM_Linear"
        SVM_Linear_model = SVC(kernel='linear', random_state=0)
        return (name, SVM_Linear_model)

    def SVM_rbf(self):
        """
        create a SVM classifier
        :return (name of the mode, QDA model):
        """
        name = "SVM_rbf"
        SVM_rbf_model = SVC(kernel='rbf', random_state=0)
        return (name, SVM_rbf_model)

    def SVM_polynomial(self):
        """
        create a SVM classifier
        :return (name of the mode, QDA model):
        """
        name = "SVM_polynomial"
        SVM_polynomial_model = SVC(kernel='poly', degree=2, random_state=0)
        return (name, SVM_polynomial_model)


    def data_spliting(self, x, y, test_size=0.2, random_state=1):
        """
        Split the data into x_train, x_test, y_train, y_test

        :param x: x (data)
        :param y: y (labels)
        :param test_size: size of test dataset
        :param random_state: 1 or 0
        :return: x_train, x_test, y_train, y_test
        """
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        return x_train, x_test, y_train, y_test

    def plotting(self, results, names, dataset_name):
        """

        :param results: result of the classifiers
        :param names: name of the model
        :param dataset_name: name of the dataset
        :return: saving Classifiers Comparison results
        """

        plt.figure(figsize=(12, 10))
        boxplot = plt.boxplot(results, patch_artist=True, labels=names)

        # fill with colors
        colors = ['pink', 'lightblue', 'lightgreen', 'lime', 'grey']
        for box, color in zip(boxplot['boxes'], colors):
            box.set(color=color)

        title = "Classifiers Comparison _ {}".format(dataset_name)
        plt.title(title)

        # saving the plots
        fname = PLOT_PATH + title + ".png"
        plt.savefig(fname, dpi=100)
        plt.close('all')


    def confusion_metrics(self, conf_matrix, accuracy_score, method_name, dataset_name):

        TP = conf_matrix[1][1]
        TN = conf_matrix[0][0]
        FP = conf_matrix[0][1]
        FN = conf_matrix[1][0]

        # calculate the sensitivity
        conf_sensitivity = (TP / (float(TP + FN)+ 0.000001))
        # calculate the specificity
        conf_specificity = (TN / (float(TN + FP) + 0.000001))
        # calculate PPV
        ppv = (TP / (float(TP + FP) + 0.000001))
        # calculate NPV
        npv = (TN / (float(TN + FN) + 0.000001))

        print("**************")
        print("Classifier: {} _ Dataset: {}".format(method_name, dataset_name))
        print("PPV:{:.2f} NPV:{:.2f} Sensitivity:{:.2f} Specificity:{:.2f}".format(ppv, npv, conf_sensitivity, conf_specificity))
        print("Accuracy Score for test_set: {:.2f} ".format(accuracy_score))

    def plot_decision_boundary(self, model, X, Y, model_name, dataset_name):

        # Set min and max values and give it some padding
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        h = 0.01

        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        x_in = np.c_[xx.ravel(), yy.ravel()]

        # Predict the function value for the whole gid
        y_pred = model.predict(x_in)
        y_pred = np.round(y_pred).reshape(xx.shape)

        # Plot the contour and training examples
        plt.contourf(xx, yy, y_pred, cmap="Pastel1")
        plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap="Pastel2")
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

        title = "Decision boundry of {} on {}".format(model_name, dataset_name)
        plt.title(title)

        # saving the plots
        fname = PLOT_PATH + title + ".png"
        plt.savefig(fname, dpi=100)
        plt.close('all')


        def storing_results(self, Results, dataset_name):
            """
            Storing the results in a csv file
            :return:
            """

            info = pd.DataFrame(Results,
                                columns=['Classifier', 'PPV', 'NPV', 'Sensitivity', 'Specificity', 'Testing_Accuracy'])
            self.store(info, dest=RESULT_PATH, name=dataset_name)

        def store(self, df, dest, name):
            """
            Storing the results as an excel file in a folder
            :param dest:
            :param name:
            :return:
            """
            path = dest + name + ".xlsx"

            df.to_excel(path)

        def grid_search_tuning(self, Models, x_train, y_train):

            for name, model in Models:
                if name == "SVM_rbf":

                    continue
                    C_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
                    gamma_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
                    degrees = range(3, 10, 2)
                    parameters = dict(gamma=gamma_range, C=C_range, degree=degrees)

                else:
                    parameters = {
                        'max_features': [7, 8, 9, 10, 12, 13, 14, 15, 20],
                        'n_estimators': [250, 300, 350, 400, 450, 500, 550],
                    }

                grid = GridSearchCV(model, parameters, refit=True, verbose=0)

                # fitting the model for grid search
                grid.fit(x_train, y_train)

                gs = GridSearchCV(estimator=model,
                                  param_grid=parameters,
                                  scoring='accuracy',
                                  cv=10,
                                  refit=True,
                                  n_jobs=-1,
                                  verbose=0)

                gs.fit(x_train, y_train)

                if name == "Random_Forest":
                    grid_search(gs.cv_results_, change='max_features')
                    title = "Progress of the GridSearch _ {}".format(name)
                    plt.title(title)
                    plt.show()

                    # saving the plots
                    fname = PLOT_PATH + title + ".png"
                    plt.savefig(fname, dpi=100)
                    plt.close('all')

                best_accuracy = gs.best_score_
                best_parameters = gs.best_params_

                print("{} Classifies: ***** ".format(name))
                print("Best Accuracy: {:.2f} %".format(best_accuracy * 100))
                print("Best Parameters:", best_parameters)


        def plotting_ROC(self):
            """
            plotting the ROC
            :param self:
            :return:
            """

        def plotting_confusion_matrix(self, model, x_test, y_test, name):

            plot_confusion_matrix(model, x_test, y_test)
            title = "Confusion Matrix _ {}".format(name)
            plt.title(title)
            plt.show()

            # saving the plots
            fname = PLOT_PATH + title + ".png"
            plt.savefig(fname, dpi=100)
            plt.close('all')



