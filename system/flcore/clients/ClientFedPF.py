from fairlearn.reductions import GridSearch, ExponentiatedGradient
from fairlearn.reductions import DemographicParity, ErrorRate, EqualizedOdds
from fairlearn.preprocessing import *
from fairlearn.postprocessing import *
from flcore.clients.clientbase import Client
from sklearn import svm, neighbors, tree
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import shap
import numpy as np
import scipy.stats as st
from sklearn.model_selection import train_test_split
import pandas as pd
import io
import requests
import seaborn as sns
import pickle
import os
import math
from pandas.api.types import CategoricalDtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import cross_val_score
from fairlearn.datasets import fetch_boston
import pulp
import math, copy
import matplotlib
import random
from fairlearn.metrics import (
                                   MetricFrame,
                                   count,
                                   false_negative_rate,
                                   false_positive_rate,
                                   selection_rate,
                              )

from torch.utils.tensorboard import SummaryWriter

class clientFedPF(Client):
        def __init__(self, args, id, train_samples, test_samples, **kwargs):
            super().__init__(args, id, train_samples, test_samples, **kwargs)
            self.model =  LogisticRegression(solver='liblinear', fit_intercept=True, penalty='l2')
            self.error = []
            self.disc = []
            self.metric = []
            self.exp = args.exp   
            self.exf = args.exf
            self.privacy_fair = args.is_fair
            self.privacy_group = args.is_privacy
            self.fair_algorithm = args.fair_algorithm
            self.global_steps = args.global_rounds
            self.writer = SummaryWriter(f'runs/experiment_{args.exn}_f{args.exf}{args.is_fair}_p{args.exp}{args.is_privacy}_d{args.dataset}/client_{id}')
          
        def train(self, count):
          # dataset
            if self.dataset == "adult":
                X_raw, Y = shap.datasets.adult()
                A = X_raw["Sex"]
                X = X_raw.drop(labels=['Sex'],axis = 1)
            elif self.dataset == "bank":
                # trainloader = self.load_train_data()
                df = pd.read_csv("./dataset/bank/UCI_Credit_Card.csv")
                df['SEX'] = df['SEX'] - 1 # 将性别转成 [0,1]形式
                df.rename(columns={'default.payment.next.month':'def_pay'}, inplace=True)
                df.rename(columns={'PAY_0':'PAY_1'}, inplace=True)
                X = df.drop(['def_pay'], axis=1)
                A = X["SEX"]
                Y = df["def_pay"]
            elif self.dataset == "compas":
                df = pd.read_csv("./dataset/compas/propublicaCompassRecividism_data_fairml.csv/propublica_data_for_fairml.csv")
                X = df.drop(['Two_yr_Recidivism'], axis=1)
                A = df["African_American"]
                Y = df["Two_yr_Recidivism"]
            X = pd.get_dummies(X)
            sc = StandardScaler()  # Normalize features by removing the mean and scaling to unit variance.
            X_scaled = sc.fit_transform(X) # Transform the training data into a standard normal distribution by converting the calculated mean and variance
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
            le = LabelEncoder() # 编码 
            Y = le.fit_transform(Y)
                    
            exp = self.exp  # 隐私
            max_trials = 1

            exp_results_error_all = []
            exp_results_fair_all = []
          # for exp in exponents:
            exp_results_error = []
            exp_results_fair = []
            global_split = int(Y.shape[0] / (self.global_steps + 1))
            for exp_2 in range(0,max_trials):
                precision_max_0=[]
                precision_max_1=[]
               # split data
                X_train, X_test, Y_train, Y_test, A_train, A_test = train_test_split(X_scaled[(count-1)*global_split:count*global_split], 
                                                                      Y[(count-1)*global_split:count*global_split], 
                                                                      A[(count-1)*global_split:count*global_split],
                                                                      test_size = 0.25,
                                                                      random_state=random.randint(1,1000),  #  在不同的客户端 该数据不相同
                                                                      stratify=Y[(count-1)*global_split:count*global_split])

                X_train, X_val, Y_train, Y_val, A_train, A_val = train_test_split(X_train, 
                                                                      Y_train, 
                                                                      A_train,
                                                                      test_size = 0.1,
                                                                      random_state=random.randint(1,1000),
                                                                      stratify=Y_train)

               # Work around indexing bug
                X_train = X_train.reset_index(drop=True)
                A_train = A_train.reset_index(drop=True)
                X_test = X_test.reset_index(drop=True)
                A_test = A_test.reset_index(drop=True)
                X_val = X_val.reset_index(drop=True)
                A_val = A_val.reset_index(drop=True)
               
                # privacy
                if self.privacy_group == True:
                    epsilon = 10**exp
                    pi = math.exp(epsilon) / (math.exp(epsilon)  +1)
                    print("for epsilong = " + str(10**exp))
                    for i in range(0,len(A_train)):
                            coin = np.random.binomial(1, pi, 1)[0] #（一次数量样本数，事件发生概率，实验次数）
                            if coin == 0:
                                A_train[i] = abs(A_train[i]-1) # 将识别类别转换
                    for i in range(0,len(A_val)):
                            coin = np.random.binomial(1, pi, 1)[0]
                            if coin == 0:
                                A_val[i] = abs(A_val[i]-1)
                    A_test_corr = copy.deepcopy(A_test)
                    for i in range(0,len(A_test_corr)):
                            coin = np.random.binomial(1, pi, 1)[0]
                            if coin == 0:
                                A_test_corr[i] = abs(A_test_corr[i]-1)


               # fairness
                if self.privacy_fair == True:
                    if self.fair_algorithm == "ExponentedGrandient":
                            step1 = ExponentiatedGradient(self.model,
                                                            constraints=EqualizedOdds(),
                                                            eps= self.exf,) # (评估器，限制条件，)
                    elif self.fair_algorithm == "GridSearch":
                            step1 = GridSearch(self.model,
                                                constraints=EqualizedOdds(),
                                                )
                    elif self.fair_algorithm == "ThresholdOptimizer":
                            step1 = ThresholdOptimizer(self.model,
                                                constraints=EqualizedOdds(),
                                                predict_method='predict_proba')
                    elif self.fair_algorithm == "CorrelationRemover":
                            step1 = CorrelationRemover(self.model,
                                                constraints=EqualizedOdds(),
                                                predict_method='predict_proba')
                        # elif self.fair_algorithm == "AdversarialFairnessClassifier":
                        #      step1 = AdversarialFairnessClassifier(self.model,
                        #                          constraints=EqualizedOdds(),
                        #                          predict_method='predict_proba')
                        # elif self.fair_algorithm == "AdversarialFairnessRegressor":
                        #      step1 = AdversarialFairnessRegressor(self.model,
                        #                          constraints=EqualizedOdds(),
                        #                          predict_method='predict_proba')
                    print("fairness constraint:{}".format(self.exf))
                    step1.fit(X_train, Y_train,
                            sensitive_features=A_train)
                    error = accuracy_score(step1.predict(X_test),Y_test)
                    print("error:{}".format(error))
                    self.model.coef_ = np.mean([model.coef_ for model in step1.predictors_], axis=0)
                    self.model.intercept_ = np.mean([model.intercept_ for model in step1.predictors_])
                    self.model.classes_ = step1.predictors_[0].classes_
                    # self.error.append(error)
                    # self.writer.add_scalar("error", error, count)
                    # disc = (max(self.measure_EO(Y_test,A_test,step1.predict(X_test) ,0),
                    #         self.measure_EO(Y_test,A_test,step1.predict(X_test) ,1)))
                    # self.disc.append(disc)
                    # self.writer.add_scalar("disc", disc, count)
                    # metrics = {
                    #             "accuracy": accuracy_score,
                    #             "precision": precision_score,
                    #             "false positive rate": false_positive_rate,
                    #             "false negative rate": false_negative_rate,
                    #             "selection rate": selection_rate,
                    #             }
                    # metric_frame = MetricFrame(metrics=metrics, y_true=Y_test, y_pred=step1.predict(X_test), sensitive_features=A_test)
                    # print(metric_frame.by_group)
                    # self.writer.add_scalars("metric", {
                    #     "accuracy_0": metric_frame.by_group['accuracy'][0],
                    #     "accuracy_1": metric_frame.by_group['accuracy'][1],
                    #     "precision_0": metric_frame.by_group['precision'][0],
                    #     "precision_1": metric_frame.by_group['precision'][1],
                    #     "false positive rate_0": metric_frame.by_group['false positive rate'][0],
                    #     "false positive rate_1": metric_frame.by_group['false positive rate'][1],
                    #     "false negative rate_0": metric_frame.by_group['false negative rate'][0],
                    #     "false negative rate_1": metric_frame.by_group['false negative rate'][1],
                    #     "selection rate_0": metric_frame.by_group['selection rate'][0],
                    #     "selection rate_1": metric_frame.by_group['selection rate'][1],
                    # }, count)
                    # self.metric.append(metric_frame.by_group)
                    # precision_max_0.append(metric_frame.by_group['precision'][0])
                    # precision_max_1.append(metric_frame.by_group['precision'][1])
                else:
                     self.model.fit(X_train, Y_train)
                     error = accuracy_score(self.model.predict(X_test),Y_test)
                    #  print("error:{}".format(error))
                    #  self.error.append(error)
                    #  self.writer.add_scalar("error", error, count)
                    #  disc = (max(self.measure_EO(Y_test,A_test,self.model.predict(X_test) ,0),
                    #         self.measure_EO(Y_test,A_test,self.model.predict(X_test) ,1)))
                    #  self.disc.append(disc)
                    #  self.writer.add_scalar("disc", disc, count)
            # print("fair:{}".format(sum(exp_results_fair)/len(exp_results_fair)))
            # print("error:{}".format(sum(exp_results_error)/len(exp_results_error)))
            # exp_results_error_all.append(exp_results_error)
            # exp_results_fair_all.append(exp_results_fair)
          # self.model = step1.predictors_[step1.best_iter_-1]
        # if self.privacy_fair == True:
            


        def measure_EO(self, Y , A, Yhat, y_value):
          # cite the paper "Fair Learning with Private Demographic Data"
          classes = len(set(A))
          Yhats = [0] * classes
          conditionals = [0] * classes
          for i in range(0,len(Y)):
               if Y[i] == y_value:
                    Yhats[A[i]] += Yhat[i]
                    conditionals[A[i]] += 1
          expectations = [0] * classes
          for i in range(0, classes):
               if conditionals[i] > 0:
                    expectations[i] = Yhats[i] / (conditionals[i]) 
               else:
                    expectations[i] = 0
               print(expectations) 
          EO_diff = []
          for i in range(0, classes):
               for j in range(i+1, classes):
                    EO_diff.append(abs(expectations[i] - expectations[j]))
          EO_violation = max(EO_diff)
          return EO_violation