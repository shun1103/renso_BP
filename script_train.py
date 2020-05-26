import sys, os
sys.path.append(os.pardir)
import argparse
import datetime
import time
import csv
#import matplotlib.pyplot as plt
#import random
import MySaveLoadFunc as MSLF
#from common import util
from common.optimizer import *
from model import TwoLayerNet
import matplotlib.pyplot as plt



#1次元時系列を学習させて再現できるかを検証
#ある程度の長さが再現できるようなら複数学習させて予測にも用いれるかも






parser = argparse.ArgumentParser(description="乱数ベクトルを記憶させる")

parser.add_argument('-hd', '--hdn', type=int, default=64, help='中間層の大きさ')
parser.add_argument('-c', '--cor', type=float, help='データの相関係数')
parser.add_argument('-p', '--ptn', type=int, help='記憶させるパターン数。最大1000')

args = parser.parse_args()

hidden_num = args.hdn
cor = args.cor
pattern_num = args.ptn

vec = []

path = './corrvec/corvecs_cor' + str(cor) + '_dim200' + '_num1000' + '.txt'

vec = MSLF.open_input_vec(path)
vec_dim = vec[0].size

print('相関係数：' + str(cor))
print('隠れ層ニューロン数：' + str(hidden_num))
print('訓練パターン数：' + str(pattern_num))
print('ベクトル次元数：' + str(vec_dim))

x_train = vec[:pattern_num]
t_train = MSLF.shiftArray(x_train, 1)

model = TwoLayerNet(input_size=vec_dim, hidden_size=hidden_num, output_size=vec_dim)
optimizer = SGD()

epoch_num = 150000
train_loss_list = []
_train_time = 0.0

loss_prev = 10000.0
break_flag = False


print("TRAIN START : " + str(datetime.datetime.now()))
t_s = time.time()

for i_epoch in range(epoch_num):
    print('epoch ' + str(i_epoch) + ' =============================================')
    grads = model.gradient(x_train, t_train)

    optimizer.update(model.params, grads)

    loss = model.loss(x_train, t_train)
    print('loss : ' + str(loss))
    if loss < 0.0001:
        break_flag = True
        break
    train_loss_list.append(loss)


t_f = time.time()
_train_time = t_f - t_s

if break_flag:
    epoch_num = i_epoch

print('TRAIN FINISH : ' + str(datetime.datetime.now()))
print('Total train epoch : ' + str(i_epoch + 1))
print('Total train time : ' + str(_train_time) + '[sec]')

# 訓練済みモデルの保存
timeID = datetime.datetime.now()
timeID_name = timeID.strftime('%m%d-%H%M%S')
dateID = datetime.datetime.today()
dateID_name = dateID.strftime('%m%d')

pickle_pardir = 'MODELS/capacity/hdn' + str(hidden_num)
os.makedirs(pickle_pardir, exist_ok=True)
output_file = os.path.join(pickle_pardir, 'model_cor' + str(cor) + '_hdn' + str(hidden_num) + '_ptn' + str(pattern_num) + '_' + timeID_name + '.dump')
MSLF.save_pickle(output_file, model)
print('Saved Model : ' + output_file)
print('Model Saved!!')

#訓練結果の保存
dateID = datetime.datetime.today()
dateID_name = dateID.strftime('%m%d')
output_dir = './TRAIN_RESULT/' + dateID_name

os.makedirs(output_dir, exist_ok=True)

body = []
header = ["epoch", "loss"]
for i, l in enumerate(train_loss_list):
    body.append([i, l])
with open(os.path.join(output_dir,'loss_cor' + str(cor) + "_hdn" + str(hidden_num) + "_ptn" + str(pattern_num) + '_' + dateID_name + '.csv'), 'w') as fo:
    writer = csv.writer(fo)
    writer.writerow(header) # ヘッダーを書き込む
    writer.writerows(body)

#訓練詳細をテキストで保存
output_dir = './TRAIN_RESULT/' + dateID_name

os.makedirs(output_dir, exist_ok=True)
body = 'date:' + timeID_name
body = body + '\n' + 'time:' + str(_train_time)
body = body + '\n' + 'epoch:' + str(i_epoch + 1)
body = body + '\n' + 'last loss:' + str(loss)

with open(os.path.join(output_dir,'train_result_cor' + str(cor) + "_hdn" + str(hidden_num) + "_ptn" + str(pattern_num) + '_' + dateID_name + '.txt'), 'w') as fo:
    fo.write(body)







"""
f = plt.figure(figsize=(10, 6))
a = f.add_subplot(111)
markers = {'train': 'o', 'test': 's'}
x = np.arange(epoch_num)
# x = np.array([i for i in range(len(train_loss_list)) if i%10==0]) # プロット点数削減
# plot_train_loss_list = [train_loss_list[i] for i in range(len(train_loss_list)) if i in x] # プロット点数削減
# plot_test_loss_list = [test_loss_list[i] for i in range(len(test_loss_list)) if i in x] # プロット点数削減
a.plot(x, train_loss_list[:epoch_num], label='train loss')
a.set_xlabel("epochs")
a.set_ylabel("loss")
# a.legend(loc='lower right')
plt.grid()
plt.show()
"""