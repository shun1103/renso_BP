# -*- coding: utf-8 -*-

from __future__ import division  # py3 "true division"

from collections import deque
import logging
import numpy as np
from queue import Queue, Empty


# If pyemd C extension is available, import it.
# If pyemd is attempted to be used, but isn't installed, ImportError will be raised in wmdistance
try:
    from pyemd import emd
    PYEMD_EXT = True
except ImportError:
    PYEMD_EXT = False

import os
import sys
import pickle
import numpy
import math
from six import string_types, integer_types, moves
#from gensim import utils, matutils
from numpy import dot, float32 as REAL, empty, memmap as np_memmap, double, array, zeros, vstack, sqrt, newaxis, ndarray, sum as np_sum, prod, argmax, divide as np_divide

logger = logging.getLogger(__name__)

def save_pickle(path, model):
    with open(path, "wb") as f_model:
        pickle.dump(model, f_model)

def load_pickle(path):
    with open(path, "rb") as f_model:
        model = pickle.load(f_model)
    return model

def save_mlps(mlps, path):
  for key in mlps.keys():
    save_pickle(path + "/" + "mlp" + key + ".dump", mlps[key])

def save_text(learn_dic, path):
  for key in learn_dic.keys():
    f = open(path + "/" + key, "w")
    for line in learn_dic[key][0]:
      f.write(line)
    f.close()

def load_vec(path):
  dic = {}
  with open(path) as f:
      line = f.readline()
      while line:
        word = line.split(" ")[0]
        vecs = map(float, line.strip("\n").split(" ")[1:])
        dic[word] = vecs
        line = f.readline()
  return dic

def make_learn_data(path1, path2, som=False):
  if os.path.exists(path1) and os.path.exists(path2):
    f1 = open(path1)
    f2 = open(path2)
  else:
    print("ファイルがありません")
    exit()
  learn_dic = {}
  line1 = f1.readline()
  line2 = f2.readline()
  while line1:
    vec1 = map(float, line1.strip("\n").split(" "))
    point1 = som.best_matching_unit(numpy.array(vec1))
    line1 = f1.readline()
    vec2 = map(float, line1.strip("\n").split(" "))
    point2 = som.best_matching_unit(numpy.array(vec2))
    point = str(point1[0]) + "_" + str(point1[1]) + "-" + str(point2[0]) + "_" + str(point2[1])
    if point not in learn_dic.keys():
      learn_dic[point] = [[line2],[[vec1, vec2]]]
    else:
      learn_dic[point][0].append(line2)
      learn_dic[point][1].append([vec1, vec2])
    line1 = f1.readline()
    line2 = f2.readline()
  f1.close()
  f2.close()
  return learn_dic

def read_relation_dic(path, reverse=False):
  dic = {}
  with open(path) as f:
    line = f.readline()
    while line:
      if reverse:
        word2 = line.split(" ")[0]
        word1 = line.strip("\n").split(" ")[1]
      else:
        word1 = line.split(" ")[0]
        word2 = line.strip("\n").split(" ")[1]
      if word1 in dic:
        if word2 not in dic[word1]:
          dic[word1].append(word2)
      else:
        dic[word1] = [word2]
      # if word2 in dic:
      #   dic[word2].append(word1)
      # else:
      #   dic[word2] = [word1]
      line = f.readline()
  return dic

def read_word2vec_dic(path):
  dic = {}
  f = open(path)
  # f = open("renso_normalized2")
  line = f.readline()
  while line:
    word = line.strip("\n").split(" ")[0]
    vecs = line.strip("\n").split(" ")[1:]
    dic[word] = numpy.array(vecs, dtype="float32")
    line = f.readline()
  f.close()
  return dic

"""
def vec2predw(vecs, word_vectors):
  # vecsは予測した出力(バッチ)。cos類似度でword_dicからword検索
  ret_wordlist = []
  ret_simlist = []
  for vec in vecs:
    max_sim = 0.0
    max_word = ""
    for k, v in word_vectors.items():
      tmp_sim = calc_cossim(vec, v)
      if tmp_sim > max_sim:
        max_sim = tmp_sim
        max_word = k
    ret_wordlist.append(max_word)
    # ret_simlist.append(max_sim)
  return ret_wordlist
  # return ret_wordlist, ret_simlist
"""
def vec2predw(vecs, word_vectors):
  # vecsは予測した出力(バッチ)。cos類似度でword_dicからword検索
  ret_wordlist = []
  ret_simlist = []
  for vec in vecs:
    similar_top_list = similar_by_word(word_vectors, vec, topn=1)
    for similar_set in similar_top_list:
      max_word = similar_set[0].encode('utf-8')
      max_sim = similar_set[1]
    ret_wordlist.append(max_word)
    ret_simlist.append(max_sim)
  # return ret_wordlist
  return ret_wordlist, ret_simlist

def vec2word(vecs, word2vec_dic):
  if vecs.ndim == 1:
    for k, v in word2vec_dic.items():
      if numpy.array_equal(v, vecs): # v, vecs : ndarray
        return k
    return "UNK"
  else: # vecsがバッチのときを想定
    ret_wordlist = []
    for vec in vecs:
      flag = False
      for k, v in word2vec_dic.items():
        if numpy.array_equal(v, vec):  # v, vec : ndarray
          ret_wordlist.append(k)
          flag = True
          break
      if not flag: ret_wordlist.append("<UNK>")
    return ret_wordlist

def calc_cossim(vec1, vec2):
    return numpy.dot(vec1, vec2) / (numpy.linalg.norm(vec1) * numpy.linalg.norm(vec2))

def calc_mean(data):
    mean = sum(data) / len(data)
    return mean

def find_diff(data):
    mean = calc_mean(data)
    diff = []
    for num in data:
        diff.append(num - mean)
    return diff

def calc_var(data):
    diff = find_diff(data)
    squared_diff = []
    for d in diff:
        squared_diff.append(d**2)
    sum_squared_diff = sum(squared_diff)
    variance = sum_squared_diff / len(data)
    return variance

def learn_data_shuffle(learn_data):
  order = numpy.random.permutation(len(learn_data[1]))
  ret_data = [[learn_data[0][i] for i in order], [learn_data[1][i] for i in order]]
  return ret_data

def shiftArray(arr, num):
    result = numpy.empty_like(arr)
    if num == 0:
        result = arr
    else:
        result[:-num] = arr[num:]
        result[-num:] = arr[:num]
    return result

def shiftList(list, num):
    num = num % len(list)
    return list[num:] + list[:num]

def addGaussianNoise(src, rate):
    # ノイズ耐性検証
    # 入力に使ってる相関ランダムベクトルは分散およそ1 (x = np.random.randn(n, dim)による部分)
    size = src.shape
    mean = 0.0
    var = 1.0
    gauss = numpy.random.normal(mean, var*rate, size)
    gauss = gauss.reshape(size)
    noisy = src + gauss
    return noisy

def add_incomplete(src, rate):
    # 欠落耐性検証
    size = src.shape
    result = src.copy()
    if len(size) == 1:
        len_size = size[0]
        mask_size = int(math.ceil(len_size*rate))
        zero_mask = numpy.random.choice(len_size, mask_size, replace=False)
        result[zero_mask] = 0.0
    else:
        for i in moves.range(size[0]):
            len_size = size[1]
            mask_size = int(math.ceil(len_size * rate))
            zero_mask = numpy.random.choice(len_size, mask_size, replace=False)
            result[i][zero_mask] = 0.0
    return result

def similar_by_word(word_vectors, word, topn=10, restrict_vocab=None, dis="cos"):
  """Find the top-N most similar words.

  Parameters
  ----------
  word : str
      Word
  topn : {int, False}, optional
      Number of top-N similar words to return. If topn is False, similar_by_word returns
      the vector of similarity scores.
  restrict_vocab : int, optional
      Optional integer which limits the range of vectors which
      are searched for most-similar values. For example, restrict_vocab=10000 would
      only check the first 10000 word vectors in the vocabulary order. (This may be
      meaningful if you've sorted the vocabulary by descending frequency.)
  dis : method of discance calculation
      cos : cosine similarity
      euc : 1 - euclidian distance
      stdeuc : 1 - standardized euclidean distance

  Returns
  -------
  list of (str, float)
      Sequence of (word, similarity).

  """
  #TODO 距離計算にコサイン類似度でなくユークリッド距離や標準ユークリッド距離もscipyの行列計算でやりたい(行列はR[1,n]*R[n,n]=R[n,n]?)
  if dis == "euc":
    pass
  elif dis == "stdeuc":
    pass
  else: # "cos"
    return word_vectors.most_similar(positive=[word], topn=topn, restrict_vocab=restrict_vocab)

def open_input_vec(path):
    vec = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        split_line = line.split()
        list = [float(index) for index in split_line[1:]]
        vec.append(list)
    vec = numpy.array(vec, dtype=numpy.float32)

    return vec

def open_mnist(path):
    vec = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        split_line = line.split(', ')
        list = [float(index) for index in split_line]
        vec.append(list)
    vec = numpy.array(vec, dtype=numpy.float32)

    return vec
