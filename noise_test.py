# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import datetime
import argparse
import glob
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import MySaveLoadFunc as MSLF


parser = argparse.ArgumentParser(description='乱数ベクトルの想起')

parser.add_argument('-hd', '--hdn', type=int, default=64, help='中間層の大きさ')
parser.add_argument('-c', '--cor', type=float, help='データの相関係数')
parser.add_argument('-p', '--ptn', type=int, help='記憶させるパターン数。最大1000')
parser.add_argument('-nr', type=float)

args = parser.parse_args()

hidden_num = args.hdn
cor = args.cor
pattern_num = args.ptn
noise_rate = args.nr
ng_flag = 0
success_len = 0

test_size = 1
sim_list = []


#print('訓練パターン数：' + str(pattern_num))
#print('隠れ層ニューロン数：' + str(hidden_num))

dateID = datetime.datetime.today()
dateID_name = dateID.strftime('%m%d')
pickle_pardir = 'MODELS/noise_80/hdn' + str(hidden_num) + '/'

load_file = pickle_pardir + 'model_cor' + str(cor) + '_hdn' + str(hidden_num) + '_ptn' + str(pattern_num) + '_*'
search_ret = glob.glob(load_file)
#print(search_ret)
input_file = search_ret[-1]
#print(input_file)
#print('Reading dump file : ' + str(input_file))
model = MSLF.load_pickle(input_file)

#TODO テスト方法の見直しとall_test_v2の実装
vec = []

path = './corrvec/corvecs_cor' + str(cor) + '_dim200' + '_num1000' + '.txt'

vec = MSLF.open_input_vec(path)
vec_dim = vec[0].size

test_size = 1
test_list = vec[:pattern_num]
test_list = MSLF.shiftArray(test_list, 1)

#print('\n==== For Normal Version Test ====')
x_test = vec[:test_size]
for i_ptn in range(pattern_num):
    #print('パターン' + str(i_ptn + 1) + ' =================================')
    t_test = test_list[i_ptn:i_ptn + test_size]
    x_test = MSLF.addGaussianNoise(x_test, noise_rate)
    pred_vec, sim = model.cos_similarity(x_test, t_test)
    if sim >= 0.95:
        if ng_flag == 0:
            success_len = len(sim_list) + 1
    elif sim < 0.95:
        ng_flag = 1

    sim_list.append(sim)
    #print(pred_vec)
    #print('コサイン類似度：' + str(sim))
    x_test = pred_vec

#print('Test Complete!')
print('noise rate : ' + str(noise_rate))
print('success_len : ' + str(success_len) + '\n')
"""
予測した乱数ベクトルを教師データとともに出力するためのコード

f = plt.figure(figsize=(12, 4.8))
a = f.add_subplot(111)
x = np.arange(200)
plt.xlim(0, 200)
plt.ylim(-5.0, 5.0)
plt.xticks(np.arange(0, 200, 10))
a.set_xlabel("dim")
a.plot(x, pred_vec[0], label='pred_vec')
a.plot(x, t_test[0], label='t_test')
a.legend(loc='upper right')
plt.grid()
plt.show()

"""

"""
#クロスエントロピーのグラフを保存
output_dir = 'fig/script/'


output_fig = 'script_hd' + str(hidden_num) + '_p' + str(pattern_num) + '_c' + str(cor) + '_' + dateIDname + '.png'

f = plt.figure(figsize=(10, 4.8))
a = f.add_subplot(111)
x = np.arange(pattern_num)
plt.xlim(0, pattern_num)
plt.ylim(0, 1.0)
plt.xticks(np.arange(0, pattern_num+1, 10))
# x = np.array([i for i in range(len(train_loss_list)) if i%10==0]) # プロット点数削減
# plot_train_loss_list = [train_loss_list[i] for i in range(len(train_loss_list)) if i in x] # プロット点数削減
# plot_test_loss_list = [test_loss_list[i] for i in range(len(test_loss_list)) if i in x] # プロット点数削減
a.plot(x, sim_list[:pattern_num])
a.set_xlabel("pattern")
a.set_ylabel("cos_similarity")
# a.legend(loc='lower right')
plt.grid()
plt.savefig(os.path.join(output_dir, output_fig))

#クロスエントロピーの値をcsvファイルに保存
output_dir = './TEST_RESULT/script/' + dateIDname
os.makedirs(output_dir, exist_ok=True)
"""
timeID = datetime.datetime.now()
timeIDname = timeID.strftime("%m%d-%H%M%S")
output_dir = "./TEST_RESULT/noize_80/hdn" + str(hidden_num) + '/cor' + str(cor)
os.makedirs(output_dir, exist_ok=True)

test_count = 5


body = []
header = ['epoch', 'loss']
for i, l in enumerate(sim_list):
    body.append([i, l])
with open(os.path.join(output_dir, 'Test_' + str(test_count) + '_noize_80_rate'+ str(noise_rate) + '_cor' + str(cor) + '_hdn' + str(hidden_num)
                                   + '_ptn' + str(pattern_num) + '_' + dateID_name + '.csv'), 'w') as fo:
    writer = csv.writer(fo)
    writer.writerow(header)
    writer.writerows(body)