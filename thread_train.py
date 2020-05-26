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


parser = argparse.ArgumentParser(description="乱数ベクトルを記憶させる")

parser.add_argument('-hd', '--hdn', type=int, default=64, help='中間層の大きさ')
parser.add_argument('-c', '--cor', type=float, help='データの相関係数')
parser.add_argument('-p', '--ptn', type=int, help='記憶させるパターン数')
parser.add_argument('-tn', type=int, help='記憶させる系列の数')

args = parser.parse_args()

hidden_num = args.hdn
cor = args.cor
pattern_num = args.ptn
thread_num = args.tn
if pattern_num % thread_num != 0:
    print('Please enter pattern_num to be divisible by thread_num')
    sys.exit()

vec = []

path = './corrvec/corvecs_cor' + str(cor) + '_dim200' + '_num1000' + '.txt'

vec = MSLF.open_input_vec(path)
vec_dim = vec[0].size

print('相関係数：' + str(cor))
print('隠れ層ニューロン数：' + str(hidden_num))
print('訓練パターン数：' + str(pattern_num))
print('訓練系列数：' + str(thread_num))
print('ベクトル次元数：' + str(vec_dim))

x_train_list = []
t_train_list = []

pattern_per_thread = int(pattern_num / thread_num)

for index in range(thread_num):
    item_x_train = vec[index * pattern_per_thread: (index + 1) * pattern_per_thread]
    item_t_train = MSLF.shiftArray(item_x_train, 1)
    x_train_list.append(item_x_train)
    t_train_list.append(item_t_train)


model = TwoLayerNet(input_size=vec_dim, hidden_size=hidden_num, output_size=vec_dim)
optimizer = SGD()

epoch_num = 50000
train_loss_list = []
_train_time = 0.0

loss_prev = 10000.0
break_flag = False


print("TRAIN START : " + str(datetime.datetime.now()))
t_s = time.time()

for i_epoch in range(epoch_num):
    print('epoch ' + str(i_epoch) + ' =============================================')
    epoch_train_list = []
    for i_thread in range(len(x_train_list)):
        grads = model.gradient(x_train_list[i_thread], t_train_list[i_thread])

        optimizer.update(model.params, grads)

        thread_loss = model.loss(x_train_list[i_thread], t_train_list[i_thread])
        epoch_train_list.append(thread_loss)

    loss_avg = sum(epoch_train_list) / len(epoch_train_list)
    print('loss : ' + str(loss_avg))
    if loss_avg < 0.0001:
        break_flag = True
        break
    train_loss_list.append(loss_avg)

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
pickle_pardir = 'MODELS'
output_file = os.path.join(pickle_pardir, 'model_cor' + str(cor) + '_hdn' + str(hidden_num) +
                           '_ptn' + str(pattern_num) + '_thread' + str(thread_num) + '_' + timeID_name + '.dump')
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
with open(os.path.join(output_dir,'loss_cor' + str(cor) + "_hdn" + str(hidden_num)
                                  + "_ptn" + str(pattern_num) + '_thread' + str(thread_num) + '_' + dateID_name + '.csv'), 'w') as fo:
    writer = csv.writer(fo)
    writer.writerow(header) # ヘッダーを書き込む
    writer.writerows(body)

#訓練詳細をテキストで保存
dateID = datetime.datetime.today()
dateID_name = dateID.strftime('%m%d')
output_dir = './TRAIN_RESULT/' + dateID_name

os.makedirs(output_dir, exist_ok=True)
body = 'date:' + timeID_name
body = body + '\n' + 'time:' + str(_train_time)
body = body + '\n' + 'epoch:' + str(i_epoch + 1)
body = body + '\n' + 'last loss:' + str(loss_avg)

with open(os.path.join(output_dir,'train_result_cor' + str(cor) + "_hdn" + str(hidden_num)
                                  + "_ptn" + str(pattern_num) + '_thread' + str(thread_num) + '_' + dateID_name + '.txt'), 'w') as fo:
    fo.write(body)







