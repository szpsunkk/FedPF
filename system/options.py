#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse   

def args_parser():  
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="mnist")
    parser.add_argument('-nb', "--num_classes", type=int, default=2)
    parser.add_argument('-m', "--model", type=str, default="dnn")
    parser.add_argument('-p', "--predictor", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=128)
    parser.add_argument('-lbse', "--batch_size_end", type=int, default=3000)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.001,
                        help="Local learning rate")
    parser.add_argument('-gr', "--global_rounds", type=int, default=2)
    parser.add_argument('-ls', "--local_steps", type=int, default=5)
    parser.add_argument('-algo', "--algorithm", type=str, default="FPFL")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-ar', "--attack_ratio", type=float, default=1,
                        help="Ratio of attacked clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=5,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    # parser.add_argument('-dp', "--privacy", type=str, default="no",
                        # help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=5)
    parser.add_argument('-eps', "--epsilon", type=float, default=10)
    parser.add_argument('-li', "--line", type=str, default='line', help="the method adding privacy: line: T, exp: exp(T) and pow: T^2")
    parser.add_argument('-in_eps', "--input_eps", type=str, default='6.2, 6.8, 7.3, 7.7, 8.08, 8.43, 8.73, 9.01, 9.29, 9.56, 9.80')
    parser.add_argument('-dpmu', "--decay_rate_mu", type=float, default=0.8)
    parser.add_argument('-dpsens', "--decay_rate_sens", type=float, default=0.5)
    parser.add_argument('-dpmuf', "--decay_rate_mu_flag", type=float, default=True)
    parser.add_argument('-dpsensf', "--decay_rate_sens_flag", type=float, default=True)
    # 
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='models')
    parser.add_argument('-dpn','--data_poison', type=str2bool, default=False,
                        help='True: data poisoning attack, False: no attack')
    parser.add_argument('-mp','--model_poison', type=str2bool, default=False,
                        help='True: model poisoning attack, False: no attack')
    parser.add_argument('-mps','--model_poison_scale', type=float, default=0.1,
                        help='scale of model poisoning attack (0.1 or 10)')
    
    # # Adversaries
    # parser.add_argument('-fa','--flag_attack', type=str2bool, default=False,
    #                     help='True: attack, False: no attack')
    # parser.add_argument('-adn','--adversaries_num', type=int, default=0,
    #                 help='the number of adversaries')
    # parser.add_argument('-adt','--adversaries_type', help='The type of adversaries', 
    #                     type=str, default='fr', choices=['lf', 're', 'vi', 'sr', 'fr', 'all'])
    # parser.add_argument('-adp','--data_poison', type=str2bool, default=False,
    #                     help='True: data poisoning attack, False: no attack')
    # parser.add_argument('-amp','--model_poison', type=str2bool, default=False,
    #                     help='True: model poisoning attack, False: no attack')
    
    # public data 
    # parser.add_argument('--num_commondata', type=float, default=1000,
    #                 help='number of public data which server has') 
    
    # # practical
    # parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.5,
    #                     help="Dropout rate for clients")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    # parser.add_argument('-ts', "--time_select", type=bool, default=False,
    #                     help="Whether to group and select clients at each round according to time cost")
    # parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
    #                     help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP
    # parser.add_argument('-bt', "--beta", type=float, default=0.0,
    #                     help="Average moving parameter for pFedMe, Second learning rate of Per-FedAvg, \
    #                     or L1 regularization weight of FedTransfer")
    # parser.add_argument('-lam', "--lamda", type=float, default=1.0,
    #                     help="Regularization weight for pFedMe and FedAMP")
    # parser.add_argument('-mu', "--mu", type=float, default=0.02,
    #                     help="Proximal rate for FedProx")
    # parser.add_argument('-K', "--K", type=int, default=5,
    #                     help="Number of personalized training steps for pFedMe")
    # parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.1,
    #                     help="personalized learning rate to caculate theta aproximately using K steps")
    # # FedFomo
    # parser.add_argument('-M', "--M", type=int, default=5,
    #                     help="Server only sends M client models to one client at each round")
    # # FedMTL
    # parser.add_argument('-itk', "--itk", type=int, default=4000,
    #                     help="The iterations for solving quadratic subproblems")
    # # FedAMP
    # parser.add_argument('-alk', "--alphaK", type=float, default=1.0, 
    #                     help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    # parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    # # APFL
    # parser.add_argument('-al', "--alpha", type=float, default=0.05)
    # parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    # # Ditto / FedRep
    # parser.add_argument('-pls', "--plocal_steps", type=int, default=1)
    # # MOON
    # parser.add_argument('-ta', "--tau", type=float, default=1.0)
    # # FedBABU
    # parser.add_argument('-fts', "--fine_tuning_steps", type=int, default=1)
    # # FedAFL
    # parser.add_argument('-alpha1', "--alpha1", type=int, default=0.75)
    # parser.add_argument('-alpha2', "--alpha2", type=int, default=0.01)
    # parser.add_argument('-alpha3', "--alpha3", type=int, default=0.1)
    
    # # Sageflow
    # parser.add_argument('--eth', type=float, default=1,
    #                     help='Eth of Eflow')
    # parser.add_argument('--delta', type=float, default=1,
    #                     help='Delta of Eflow')
    
    # # FedMFG
    # parser.add_argument('-ga', "--gamma", type=float, default=0.01)
    # parser.add_argument('-Thr', "--th   rethold", type=int, default=0)
    
    # FedAvg_FedPF
    parser.add_argument('-exp','--exp', type=float, default=1,
                        help='privacy constraints')
    parser.add_argument('-exf','--exf', type=float, default=0.01,
                        help='fairness constraints')
    parser.add_argument('-fair', "--is_fair", type=bool, default=False,
                        help="use the fairness algorithm")
    parser.add_argument('-privacy', "--is_privacy", type=bool, default=False,
                        help="use the privacy algorithm")
    parser.add_argument('-fair_al', "--fair_algorithm", type=str, default='ExponentedGrandient') 
    
    args = parser.parse_args()
    
    return args

def str2bool(v):
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
