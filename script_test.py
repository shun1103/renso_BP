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
parser.add_argument('-tn')

args = parser.parse_args()

hidden_num = args.hdn
cor = args.cor
pattern_num = args.ptn
test_count = args.tn

test_size = 1
sim_list = []

ng_flag = 0
success_len = 0



print('訓練パターン数：' + str(pattern_num))
print('隠れ層ニューロン数：' + str(hidden_num))
print('相関係数 : ' + str(cor))

dateID = datetime.datetime.today()
dateID_name = dateID.strftime('%m%d')
pickle_pardir = 'MODELS/capacity_all/hdn' + str(hidden_num) + '/'

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
    pred_vec, sim = model.cos_similarity(x_test, t_test)
    if sim >= 0.97:
        if ng_flag == 0:
            success_len = len(sim_list) + 1
    elif sim < 0.97:
        ng_flag = 1
    sim_list.append(sim)
    #print(pred_vec)
    #print('コサイン類似度：' + str(sim))
    x_test = pred_vec

#print('Test Complete!')
print('success_len : ' + str(success_len) + '\n')


timeID = datetime.datetime.now()
timeIDname = timeID.strftime("%m%d-%H%M%S")
output_dir = './TEST_RESULT/capacity_all_97/Test_' + str(test_count) + '/hdn' + str(hidden_num)
os.makedirs(output_dir, exist_ok=True)



body = []
header = ['epoch', 'loss']
for i, l in enumerate(sim_list):
    body.append([i, l])
with open(os.path.join(output_dir, 'capa_all_cossim_cor' + str(cor) + '_hdn' + str(hidden_num)
                                   + '_ptn' + str(pattern_num) + '_' + dateID_name + '.csv'), 'w') as fo:
    writer = csv.writer(fo)
    writer.writerow(header)
    writer.writerows(body)

readme_dir = './TEST_RESULT/capacity_all_97/Test_' + str(test_count) + '_Readme/hdn' + str(hidden_num)
os.makedirs(readme_dir, exist_ok=True)

Readme_body = '記憶パターン数：' + str(pattern_num) + '\n'
Readme_body = Readme_body + 'success_len : ' + str(success_len) + '\n'
Readme_body = Readme_body + '想起成功率：' + str(success_len/pattern_num) + '\n\n'

with open(os.path.join(readme_dir, 'capa_all_cossim_cor' + str(cor) + '_hdn' + str(hidden_num)+ '_' + dateID_name + '_Readme.txt'), 'a') as fo:
    fo.write(Readme_body)

