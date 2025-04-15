import time
from flcore.clients.ClientFedPF import clientFedPF
from flcore.servers.serverbase import Server
from sklearn.linear_model import LogisticRegression
from threading import Thread
import numpy as np
import shap
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from fairlearn.metrics import (
                                   MetricFrame,
                                   count,
                                   false_negative_rate,
                                   false_positive_rate,
                                   selection_rate,
                              )
from torch.utils.tensorboard import SummaryWriter

class FedFP(Server):
     def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients_sk(args, clientFedPF)  # 设置client对象
        self.selected_client_id = []
        self.attack_client_id = []
        self.exp = args.exp
        self.exf = args.exf
     #    self.fair_algorithm = args.fair_algorithm
        self.privacy_fair = args.is_fair
        self.privacy_group = args.is_privacy
        self.exn = args.exn

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.error = []
        self.disc = []
        self.metric = []
        self.writer = SummaryWriter(f'runs/experiment_{args.exn}_f{args.exf}{args.is_fair}_p{args.exp}{args.is_privacy}_d{args.dataset}/server')


     def train(self):
          # print(self.global_model)
          # attacked_clients = []
          # count = 0
          train_loss, train_accuracy = [], []
          self.selected_clients = self.select_clients()
          self.clients = self.selected_clients
          # attacked_clients = self.select_attack_clients()
          # for client in self.selected_clients:
          #      if client in attacked_clients:
          #           self.attack_client_id.append(client.id)
          #      else:
          #           self.selected_client_id.append(client.id)
          for i in range(self.global_rounds+1):
               print("global round", i)
               s_t = time.time()
               for client in self.selected_clients:
                    client = client.train(i+1)
            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]
               self.aggregation_global()
               self.evaluate_FP(i+1)
               self.Budget.append(time.time() - s_t)
               print('-'*25, 'time cost', '-'*25, self.Budget[-1])

          print("\nBest global accuracy.")
          print("\nAverage time cost per round.")
          print(sum(self.Budget[1:])/len(self.Budget[1:]))
          # self.save_results()


     def aggregation_global(self):
          localmodels = [client.model for client in self.selected_clients]
          global_params = np.mean([model.coef_ for model in localmodels], axis=0)
          global_intercept = np.mean([model.intercept_ for model in localmodels])
          self.global_model = LogisticRegression(solver='liblinear', fit_intercept=True, penalty='l2')
          self.global_model.coef_ = global_params
          self.global_model.intercept_ = global_intercept
          self.global_model.classes_ = localmodels[0].classes_
          # print("全局模型的系数：", self.global_model.coef_)
          # print("全局模型的节距：", self.global_model.intercept_)
          for client in self.selected_clients:
               client.model.coef_ = self.global_model.coef_
               client.model.intercept_ = self.global_model.intercept_
     
     def evaluate_FP(self, count):  # 服务器端评估公平性
          
          # X_raw, Y = shap.datasets.adult()
          # A = X_raw["Sex"]
          # X = X_raw.drop(labels=['Sex'],axis = 1)
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
          sc = StandardScaler()  #通过移除平均值并缩放至单位方差来标准化特征。
          X_scaled = sc.fit_transform(X) # 通过计算出来的均值和方差转换训练数据，把数据转换成标准的正态分布
          X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
          le = LabelEncoder() # 编码 
          Y = le.fit_transform(Y)
          global_split = int(Y.shape[0] / (self.global_rounds + 1))
          X_train, X_test, Y_train, Y_test, A_train, A_test = train_test_split(X_scaled, 
                                                                           Y, 
                                                                           A,
                                                                           test_size = 0.25,
                                                                           random_state=random.randint(1,1000),  #  在不同的客户端 该数据不相同
                                                                           stratify=Y)

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
          self.error.append(accuracy_score(self.global_model.predict(X_test),Y_test)) 
          self.disc.append((max(self.measure_EO(Y_test,A_test,self.global_model.predict(X_test) ,0),
                         self.measure_EO(Y_test,A_test,self.global_model.predict(X_test) ,1))))
          
          metrics = {
                    "accuracy": accuracy_score,
                    "precision": precision_score,
                    "false positive rate": false_positive_rate,
                    "false negative rate": false_negative_rate,
                    "selection rate": selection_rate,
                    "count": self.count_metric,
                    }
          y_pred = self.global_model.predict(X_test)
          metric_frame = MetricFrame(metrics=metrics, y_true=Y_test, y_pred=y_pred, sensitive_features=A_test)
          # self.metric.append(metric_frame.by_group)
          self.writer.add_scalars("metric", {
                      "accuracy_0": metric_frame.by_group['accuracy'][0],
                      "accuracy_1": metric_frame.by_group['accuracy'][1],
                      "precision_0": metric_frame.by_group['precision'][0],
                      "precision_1": metric_frame.by_group['precision'][1],
                      "false positive rate_0": metric_frame.by_group['false positive rate'][0],
                      "false positive rate_1": metric_frame.by_group['false positive rate'][1],
                      "false negative rate_0": metric_frame.by_group['false negative rate'][0],
                      "false negative rate_1": metric_frame.by_group['false negative rate'][1],
                      "selection rate_0": metric_frame.by_group['selection rate'][0],
                      "selection rate_1": metric_frame.by_group['selection rate'][1],
                }, count)
          self.writer.add_scalar("error", self.error[-1], count)
          self.writer.add_scalar("disc", self.disc[-1], count)
          ax = metric_frame.by_group.plot.bar(
                                        subplots=True,
                                        layout=[3, 3],
                                        legend=False,
                                        figsize=[12, 8],
                                        # title="Show all metrics",
                                        )

          figure = ax[0, 0].get_figure()
          # figure.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.2, hspace=0.3)
          figure.savefig(f'runs/experiment_{self.exn}_f{self.exf}{self.privacy_fair}_p{self.exp}{self.privacy_group}_d{self.dataset}/g-{count}-fairness_metric_plot.png', dpi = 300)
          print(metrics)
          
     def count_metric(self, y_true, y_pred):
          return count(y_true, y_pred)
          
     def measure_EO(self, Y , A, Yhat, y_value):
          '''Cite paper
          '''
          classes = len(set(A))
          Yhats = [0] * classes
          conditionals = [0] * classes
          for i in range(0,len(Y)):
               if Y[i] == y_value:
                    Yhats[A[i]] += Yhat[i]
                    conditionals[A[i]] += 1
          expectations = [0] * classes
          for i in range(0, classes):
               expectations[i] = Yhats[i] / (conditionals[i]) 
          EO_diff = []
          for i in range(0, classes):
               for j in range(i+1, classes):
                    EO_diff.append(abs(expectations[i] - expectations[j]))
          EO_violation = max(EO_diff)
          return EO_violation

