import matplotlib.pyplot as pt
import pandas as pd
import numpy as np
path_test = "/home/skk/FL/PFL/PFL-main/results/adult/Local/algo(Local)-N(5)-A(1)-dp(False)-mp(False)-model_poison_sacle(0.1)-join_ratio(1)_privacy(no)_dp_sigma(1)/acc_adult_Local_test_0.csv"
path_acc_black = "/home/skk/FL/PFL/PFL-main/results/adult/Local/algo(Local)-N(5)-A(1)-dp(False)-mp(False)-model_poison_sacle(0.1)-join_ratio(1)_privacy(no)_dp_sigma(1)/acc_black_adult_Local_test_0_dp_1.csv"
path_acc_white = "/home/skk/FL/PFL/PFL-main/results/adult/Local/algo(Local)-N(5)-A(1)-dp(False)-mp(False)-model_poison_sacle(0.1)-join_ratio(1)_privacy(no)_dp_sigma(1)/acc_white_adult_Local_test_0_dp_1.csv"
df_test = pd.read_csv(path_test, index_col=1)
print(df_test)
fig = pt.figure()
ax = fig.add_subplot(111)
ax.plot(df_test.index,df_test.iloc[:,0], color="crimson", label="loss")
pt.title("test loss")
pt.xlabel('Rounds', fontsize=13)
pt.ylabel('loss', fontsize=13)
pt.legend()
pt.savefig("/home/skk/FL/PFL/PFL-main/results/adult/Local/algo(Local)-N(5)-A(1)-dp(False)-mp(False)-model_poison_sacle(0.1)-join_ratio(1)_privacy(no)_dp_sigma(1)/a.jpg")