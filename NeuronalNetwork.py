
import yaml
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_validate, cross_val_score, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import  MLPClassifier
from sklearn.model_selection import KFold
import joblib
import numpy as np

with open('NNconfig.yaml', 'r') as file:
    params_model = yaml.safe_load(file)

class RedNeuronal(object):
    def __init__(self,data,target,SEED):
        self.data=data
        self.target= target
        self.SEED= SEED

    def Models_score_cv(self,clf, X, y):
        scores = cross_val_score(clf, X, y, cv=5, scoring='neg_mean_absolute_percentage_error')
        clf.fit(X, y)
        print("Score: {:.10f}".format((1 + np.mean(scores)) * 100))
        plt.clf()
        plt.plot(range(len(y)), clf.predict(X), "y-")
        plt.bar(range(len(y)), y)
        plt.show()
        return (1 + np.mean(scores)) * 100

    def Models_score_base(self,clf, X, y):
        clf.fit(X,y)
        score = clf.score(X, y)
        return score

    def Models_Evaluation(self,clf,X_train, X_test, y_train, y_test):
        # Evaluación de modelos
        clf.fit(X_train,y_train)
        y_pred= clf.predict(X_test)
        mcknn=confusion_matrix(y_test, y_pred)
        print("Results Model")
        print("Confusion Matrix: {}".format(mcknn))
        print("Report of Clasification Indicator")
        print(classification_report(y_test,y_pred))
        bnd= accuracy_score(y_test,y_pred)
        return bnd
    def Generate_Neuronal_Model_KNFold(self,params,N_SP,path_file):
        """
        Generates a neural model using K-Fold Cross Validation.

        Args:
            params (dict): A dictionary of parameters for the MLP classifier. 
                        It should contain the following keys: 'activation', 'hidden_layer_sizes', 
                        'learning_rate_init', 'max_iter', 'solver'.
            N_SP (int): The number of splits for K-Fold cross validation.
            path_file (str): The path to the file containing the data to be used.

        Returns:
            None. The method fits the neural model and saves the trained model in the class attribute.

        Raises:
            ValueError: If `params` does not contain the necessary keys.
            FileNotFoundError: If `path_file` does not point to an existing file.
        """
        X = self.data.drop(self.target, axis=1)
        y = self.data[self.target].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=self.SEED)
        # Transform data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        # Create Model with 2 capas
        clf = MLPClassifier(**params, random_state=self.SEED)
        ft=self.K_Fold(clf,N_SP,X_train,y_train,path_file) # This objetc contained all inforamtion about models generated in Kfolds
        best_acurracy=-99
        acum_acurracy=0
        average_acurracy=0
        k=0
        savedir= os.path.join(os.getcwd(),path_file)
        file_scaler= 'scaler.pkl'
        file_scaler = os.path.join(savedir, file_scaler)
        joblib.dump(scaler,file_scaler)
        model_return=[]
        for i ,clf_i in enumerate(ft):
            print("Start Results Model  :", i , " Fold")
            clft= joblib.load(clf_i) # This model was trained
            y_pred=clft.predict(X_test)
            labels = np.unique(np.concatenate((y_test, y_pred)))
            mcknn = confusion_matrix(y_test, y_pred, labels=labels)
            #mcknn=confusion_matrix(y_test, y_pred)
            print("Confusion Matrix: {}".format(mcknn))
            print("Report of Clasification Indicator")
            print(classification_report(y_test, y_pred))
            print("End Results Model  :", i , " Fold")
            print("##############################################")
            current_acurracy=accuracy_score(y_test,y_pred)
            if current_acurracy > best_acurracy:
                best_acurracy=current_acurracy
                acum_acurracy+=best_acurracy
                k=i
        average_acurracy=acum_acurracy/N_SP        
        best_model= joblib.load(ft[k])
        model_return.append(best_model)
        model_return.append(scaler)
        model_return.append(k)
        # The best model is the model with the best accuracy
        print("The Best Model is  Model ", k , " with a acurracy_score of : ", best_acurracy )
        print("The average acurracy_score is : ", average_acurracy)
        return model_return

    def Generate_Neuronal_Model(self,activation,neu1,neu2):
        # This code serves to fit model with activation function, neurons for each level
        X = self.data.drop(self.target, axis=1)
        y = self.data[self.target].ravel()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=self.SEED)
        # Transform data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        # Create Model with 2 capas
        hidden_layer_sizes = (neu1, neu2)
        clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=10000, random_state=42, learning_rate_init=0.1)
        clf.fit(X_train,y_train)
        mcknn=confusion_matrix(y_test, clf.predict(X_test))
        print("Results Model")
        print("Confusion Matrix: {}".format(mcknn))
        print("Report of Clasification Indicator")
        print(classification_report(y_test,clf.predict(X_test)))
        return clf,scaler
    def Generate_Prediction(self,clf,scaler,data_test):
        x_test = scaler.transform(data_test)
        y_pred= clf.predict(x_test)
        return y_pred
    def K_Fold(self,clf,N_SP,X_train,y_train,path_file):
        kf= KFold(n_splits=N_SP,shuffle= True)
        kf.get_n_splits(X_train)
        savedir= os.path.join(os.getcwd(),path_file)
        models_name=[]
        for i, (train_index, test_index) in enumerate(kf.split(X_train)):
            x_tr = X_train[train_index] # Data X of training process
            y_tr = y_train[train_index] # Data y of training process
            filename= 'model'+ str(i)+'.pkl'
            filename = os.path.join(savedir, filename)
            models_name.append(filename)
            clf.fit(x_tr,y_tr)
            joblib.dump(clf,filename)
        return models_name

    def find_best_Hyperparamters(self,find_method_=0,n_cap0=0,n_cap1=2,n_cap2=2):
        """
        This function performs hyperparameter tuning for an MLPClassifier using GridSearchCV or RandomizedSearchCV.

        Parameters:
            n_cap0 (int): The lower limit for the number of neurons in the first and second hidden layer to consider.
            n_cap1 (int): The upper limit for the number of neurons in the first hidden layer to consider.
            n_cap2 (int): The upper limit for the number of neurons in the second hidden layer to consider.
            find_method_ (int): 0 for GridSearchCV, 1 for RandomizedSearchCV.

        Returns:
            dict: A dictionary containing the best hyperparameters found.

        Usage:
            best_params = object.find_best_Hyperparamters(10, 10)

        This will perform hyperparameter tuning considering all combinations of 2 to 10 neurons in the first hidden layer and 2 to 10 neurons in the second hidden layer.

        The returned dictionary will have the following structure:
            {
                'hidden_layer_sizes': (best number of neurons in first hidden layer, best number of neurons in second hidden layer),
                'solver': best solver,
                'activation': best activation function,
                'max_iter': best maximum number of iterations,
                'learning_rate_init': best initial learning rate
        }
        """
        X = self.data.drop(self.target, axis=1)
        y = self.data[self.target].to_numpy()
        print(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params_model['test_size'], random_state=self.SEED)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        layers=[]
        for i in range(n_cap0,n_cap1):
            for j in range(n_cap0,n_cap2):
                layers.append((i,j))
        params_grid={"hidden_layer_sizes":layers,
                    "solver":params_model['solver'],
                    "activation":params_model['activation'],
                    "max_iter": params_model['max_iter_i'],
                    "learning_rate_init":params_model['learning_rate_init']}
        print(params_grid)
        clf= MLPClassifier(random_state=self.SEED)
        if find_method_==0:
            print("Grid Search")
            clf= GridSearchCV(clf,params_grid,cv=params_model['cross_validation'],n_jobs=-1)
        else:
            print("Randomized Search")
            clf= RandomizedSearchCV(clf,params_grid,cv=params_model['cross_validation'],n_iter=params_model['n_iter'],n_jobs=-1)   
        clf.fit(X_train,y_train)
        return clf.best_params_

    def Indetify_Hyperparamters_Analysis(self,n_cap1, n_cap2):
        # This code serve to identify how many hidden_layers_sizes needs for each level and testing what activation function we needs
        X = self.data.drop(self.target, axis=1)
        y = self.data[self.target].ravel()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=self.SEED)
        # Transform data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        best_score = -1
        best_activation = ""
        best_hidden_layer = []

        for activation in ['logistic', 'relu', 'identity', 'tanh']:
            for capa_1 in range(1, n_cap1):
                for capa_2 in range(1, n_cap2):
                    hidden_layer_sizes = (capa_1, capa_2)
                    clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=10000, random_state=42, learning_rate_init=0.1)
                    print("With {}".format(activation))
                    print("Layer sizes: {}".format(hidden_layer_sizes))
                    cur_score= self.Models_score_base(clf, X_train, y_train)
                    #cur_score = self.Models_Evaluation(clf,X_train, X_test, y_train, y_test)
                    if cur_score > best_score:
                        best_score = cur_score
                        best_activation = activation
                        best_hidden_layer = hidden_layer_sizes
        print("The Best Parameters is:")
        print(best_activation)
        print(best_hidden_layer)
        clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=best_hidden_layer, activation=best_activation, max_iter=10000, random_state=42, learning_rate_init=0.1)
        cur_score = self.Models_Evaluation(clf,X_train, X_test, y_train, y_test)
        return best_activation,best_hidden_layer
