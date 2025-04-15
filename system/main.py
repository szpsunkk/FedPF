#!/usr/bin/env python
import copy
import torch
import os
import time
import warnings
import numpy as np
from torch.nn.functional import dropout
import torchvision


# from flcore.servers.serveravg import FedAvg
# from flcore.servers.serveravg_f import FedAvg_f # FedAvg_f: FedAvg with fairness
from system.flcore.servers.serveravg_fedfp import FedFP
# from flcore.servers.serverpFedMe import pFedMe
# from flcore.servers.serverperavg import PerAvg
# from flcore.servers.serverprox import FedProx
# from flcore.servers.serverfomo import FedFomo
# from flcore.servers.serveramp import FedAMP
# from flcore.servers.servermtl import FedMTL
# from flcore.servers.serverlocal import Local
# from flcore.servers.serverper import FedPer
# from flcore.servers.serverapfl import APFL
# from flcore.servers.serverditto import Ditto
# from flcore.servers.serverrep import FedRep
# from flcore.servers.serverphp import FedPHP
# from flcore.servers.serverbn import FedBN
# from flcore.servers.serverrod import FedROD
# from flcore.servers.serverproto import FedProto
# from flcore.servers.serverdyn import FedDyn
# from flcore.servers.servermoon import MOON
# from flcore.servers.serverbabu import FedBABU
# from flcore.servers.serverRFFL import FedRFFL
# from flcore.servers.serversageflow import Sageflow
# from flcore.servers.serverfedmfg import FedMFG
# from flcore.servers.serverkrum import Krum
# from flcore.servers.servermultikrum import Multi_Krum
# from flcore.servers.serverfpfl import FedFPFL
# from flcore.servers.serverSPF_Pre import FedSPF_Pre
# from flcore.servers.serverSPF_Imp_LDP import FedSPF_Imp_LDP
# from flcore.servers.serverSPF_Imp_DP import FedSPF_Imp_DP
# from flcore.servers.serverafl import AFL


from flcore.trainmodel.models import *

from flcore.trainmodel.bilstm import BiLSTM_TextClassification
# from flcore.trainmodel.resnet import resnet18 as resnet
from flcore.trainmodel.alexnet import alexnet
from flcore.trainmodel.mobilenet_v2 import mobilenet_v2
from system.utils.result_utils import average_data, print_par
from system.utils.mem_utils import MemReporter
from options import args_parser
warnings.simplefilter("ignore")
torch.manual_seed(0)

# hyper-params for Text tasks
vocab_size = 98635
max_len=200
hidden_dim=32

def run(args):

    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "mlr":
            if args.dataset == "mnist" or args.dataset == "fmnist":
                args.model = Mclr_Logistic(1*28*28, num_classes=args.num_classes).to(args.device)
            elif args.dataset == "cifar10" or args.dataset == "cifar100":
                args.model = Mclr_Logistic(3*32*32, num_classes=args.num_classes).to(args.device)
            else:
                args.model = Mclr_Logistic(12, num_classes=args.num_classes).to(args.device)

        elif model_str == "cnn":
            if args.dataset == "mnist" or args.dataset == "fmnist":
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif args.dataset == "cifar10" or args.dataset == "cifar100":
                # args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
                args.model = CifarNet(num_classes=args.num_classes).to(args.device)
            elif args.dataset[:13] == "Tiny-imagenet" or args.dataset[:8] == "Imagenet":
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)


        elif model_str == "dnn": # non-convex
            if args.dataset == "mnist" or args.dataset == "fmnist":
                args.model = DNN(1*28*28, 50, num_classes=args.num_classes).to(args.device)
            elif args.dataset == "cifar10" or args.dataset == "cifar100":
                args.model = DNN(3*32*32, 100, num_classes=args.num_classes).to(args.device)
            # elif args.dataset == 'adult':
            #     args.model = DNN(12,100,num_classes=args.num_classes).to(args.device)
            elif args.dataset == "adult":
                args.model = Adult_Model(input_dim=12, hide_dim=100, output_dim=2).to(args.device) # 输入参数维度
                
            elif args.dataset == "bank":
                args.model = Bank_Model(input_dim=22, hide_dim=100, output_dim=2).to(args.device) # 输入参数维度
            elif args.dataset == "compas":
                args.model = Compas_Model(input_dim=60, hide_dim=100, output_dim=2).to(args.device) # 输入参数维度
            # elif args.dataset == "smart_grid":
            #     args.model = Smart_Grid_Model(input_dim=12, hide_dim=200, output_dim=2).to(args.device) # 输入参数维度
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)
        
        elif model_str == "resnet":
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)
            
            # args.model = torchvision.models.resnet18(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
            
            # args.model = resnet18(num_classes=args.num_classes, has_bn=True, bn_block_num=4).to(args.device)

        elif model_str == "alexnet":
            args.model = alexnet(pretrained=False, num_classes=args.num_classes).to(args.device)
            
            # args.model = alexnet(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
            
        elif model_str == "googlenet":
            args.model = torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes).to(args.device)
            
            # args.model = torchvision.models.googlenet(pretrained=True, aux_logits=False).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "mobilenet_v2":
            args.model = mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)
            
            # args.model = mobilenet_v2(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
            
        elif model_str == "lstm":
            args.model = LSTMNet(hidden_dim=hidden_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(args.device)

        elif model_str == "bilstm":
            args.model = BiLSTM_TextClassification(input_size=vocab_size, hidden_size=hidden_dim, output_size=args.num_classes, 
                        num_layers=1, embedding_dropout=0, lstm_dropout=0, attention_dropout=0, 
                        embedding_length=hidden_dim).to(args.device)

        elif model_str == "fastText":
            args.model = fastText(hidden_dim=hidden_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(args.device)

        elif model_str == "TextCNN":
            args.model = TextCNN(hidden_dim=hidden_dim, max_len=max_len, vocab_size=vocab_size, 
                            num_classes=args.num_classes).to(args.device)
        else:
            pass
    # FedPF algorithm
    server = FedFP(args, i)
    server.train()
    time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()
    args = args_parser()
    args.algorithm = "FedFP"  # FedAvg, FedProx, FedFomo, FedAMP, FedMTL, PerAvg, pFedMe, Local
    args.dataset = "compas" # adult, bank, compas
    args.model = "dnn"
    # args.num_classes = 12
    # args.batch_size = 128
    # args.privacy = 'no'
    # args.privacy = 'time'
    args.global_rounds = 200
    args.exn = 1  # experiment number
    args.is_fair = True
    args.is_privacy = False
    args.exp = 1.0 # -2, -1, 0, 1, 2
    args.exf = 1.0 # 0.01, 0.1, 1.0, 10.0, 100.0

    print(os.getcwd())
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"
    print_par(args)
    run(args)
    print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")
