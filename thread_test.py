# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import datetime
import argparse
import glob
import csv
import numpy as np
import MySaveLoadFunc as MSLF


parser = argparse.ArgumentParser(description='乱数ベクトルの想起')

parser.add_argument('-hd', '--hdn', type=int, default=64, help='中間層の大きさ')
parser.add_argument('-c', '--cor', type=float, help='データの相関係数')
parser.add_argument('-p', '--ptn', type=int, help='記憶させるパターン数。最大1000')
parser.add_argument('-tn', type=int, help='記憶させる系列の数')

args = parser.parse_args()

hidden_num = args.hdn
cor = args.cor
pattern_num = args.ptn
thread_num = args.tn

test_size = 1
sim_list = []
success_len_list = []
ng_flag = 0

print('訓練パターン数：' + str(pattern_num))
print('隠れ層ニューロン数：' + str(hidden_num))

pickle_pardir = 'MODELS/thread_3/'
load_file = pickle_pardir + 'model_cor' + str(cor) + '_hdn' + str(hidden_num) \
            + '_ptn' + str(pattern_num) + '_thread' + str(thread_num)+ '_*'
search_ret = glob.glob(load_file)
print(search_ret)
input_file = search_ret[0]
print(input_file)
print('Reading dump file : ' + str(input_file))
model = MSLF.load_pickle(input_file)

vec = []

path = './corrvec/corvecs_cor' + str(cor) + '_dim200' + '_num1000' + '.txt'

vec = MSLF.open_input_vec(path)

x_test_list = []
t_test_list = []

pattern_per_thread = int(pattern_num / thread_num)

for index in range(thread_num):
    item_x_test = vec[index * pattern_per_thread: index * pattern_per_thread + 1]
    item_t_test = vec[index * pattern_per_thread: (index + 1) * pattern_per_thread]
    item_t_test = MSLF.shiftArray(item_t_test, 1)
    x_test_list.append(item_x_test)
    t_test_list.append(item_t_test)

test_size = 1

dateID = datetime.datetime.now()
dateID_name = dateID.strftime("%m%d-%H%M%S")
output_dir = "./TEST_RESULT"

print('\n==== For Normal Version Test ====')

for i_thread in range(len(x_test_list)):
    x_test = x_test_list[i_thread]
    test_list = t_test_list[i_thread]
    sim_thread = []
    for i_ptn in range(pattern_per_thread):
        t_test = test_list[i_ptn:i_ptn + test_size]
        pred_vec, sim = model.cos_similarity(x_test, t_test)
        if sim >= 0.95:
            if ng_flag == 0:
                success_len = len(sim_thread) + 1
        elif sim < 0.95:
            ng_flag = 1
        sim_thread.append(sim)
        x_test = pred_vec
    success_len_list.append(success_len)
    sim_list.append(sim_thread)
    ng_flag = 0


print('success_len : ')
print(np.array(sim_list).shape)
[print(item) for item in success_len_list]
print('Test Complete!')


timeID = datetime.datetime.now()
timeIDname = timeID.strftime("%m%d-%H%M%S")
output_dir = './TEST_RESULT/thread_3/'
os.makedirs(output_dir, exist_ok=True)


for i_thread in range(thread_num):
    body = []
    thread_count = ['thread', i_thread]
    header = ['epoch', 'loss']
    for i, l in enumerate(sim_list[i_thread]):
        body.append([i, l])
    with open(os.path.join(output_dir, 'thread_3_cossim_cor' + str(cor) + '_hdn' + str(hidden_num)
                                   + '_ptn' + str(pattern_num) + '_' + dateID_name + '.csv'), 'a') as fo:
        writer = csv.writer(fo)
        writer.writerow(thread_count)
        writer.writerow(header)
        writer.writerows(body)

readme_dir = './TEST_RESULT/thread_3/'
os.makedirs(readme_dir, exist_ok=True)

for i_thread in range(len(success_len_list)):
    Readme_body = 'thread : ' + str(i_thread + 1) + '\n'
    Readme_body = Readme_body +  '系列の長さ：' + str(pattern_per_thread) + '\n'
    Readme_body = Readme_body + 'success_len : ' + str(success_len_list[i_thread]) + '\n'
    Readme_body = Readme_body + '想起成功率：' + str(success_len_list[i_thread] / pattern_per_thread) + '\n\n'

    with open(os.path.join(readme_dir, 'thread_3_cossim_cor' + str(cor) + '_hdn' + str(
            hidden_num) + '_ptn' + str(pattern_num) + '_' + dateID_name + '_Readme.txt'), 'a') as fo:
        fo.write(Readme_body)