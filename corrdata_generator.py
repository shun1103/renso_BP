# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

"""
# 力技 ほとんど相関係数は0.25以下に収まる
dim = 200
lim_break = 500
vec_total = 2
max_cor_whole = 0.0

for epoch in xrange(lim_break):
    flag = True
    vecs = []
    for i in xrange(vec_total):
        a = np.random.randn(dim)
        vecs.append(a)

    x = np.asarray(vecs, dtype='float32')
    cor = np.corrcoef(x)

    max_cor = 0.0

    for i in xrange(vec_total):
        for j in xrange(i+1, vec_total):
            # print cor[i, j]
            # if not 0.1 < cor[i, j] < 0.2 and not -0.2 < cor[i, j] < -0.1:
            #     flag = False
            #     break
            if max_cor < abs(cor[i, j]):
                max_cor = cor[i, j]
    # if flag:
    #     print epoch + 1
    #     print cor
    #     break
    print max_cor
    if max_cor_whole < max_cor:
        max_cor_whole = max_cor

print "max of all: ", max_cor_whole

# if epoch >= lim_break-1:
#     print "No found"
"""

def plot_result_data(z, cor):
    output_dir = "./"
    fig, ax = plt.subplots()
    ax.scatter([z[0, :]], [z[1, :]], s=1, color='red', label='Z_2')
    ax.scatter([z[0, :]], [z[2, :]], s=1, color='blue', label='Z_3')
    ax.set_xlabel('Z_1')
    ax.set_ylabel('Z_2, Z_3')
    ax.legend()
    # fig.savefig(os.path.join(output_dir, 'cor' + str(cor) + '.png'), bbox_inches='tight')
    plt.show()
    plt.close()

def save_vecs(z, filename):
    fo = open(filename, "w")
    for i, vec in enumerate(z):
        fo.write("vec" + str(i) + " ")
        for item in vec:
            fo.write(str(item) + ' ')
        fo.write('\n')
    fo.close()


# Set parameters
n = 1000 # The number of random numbers. number of data samples
dim = int(200) # Size of the vector (one vector is one sample)

# make Correlation matrix
req_cor = 0.9
mask = np.ma.make_mask(np.eye(n))
r_in_ndarr = req_cor * np.ones((n, n))
r_in_ndarr[mask] = 1.0
r_in = np.mat(r_in_ndarr)
print(r_in)

# Generate correlated random numbers
l = np.linalg.cholesky(r_in)
x = np.random.randn(n, dim)
z = l * x

# Calculate stats
cov = np.cov(z)
r_out = np.corrcoef(z)
print("covariance matrix:\n{}\n".format(cov))
print("correlation matrix:\n{}\n".format(r_out))
print(r_out.shape)

# Plot histgram of correlation coefficient
# plot_hist_cor(r_out)

# Plot results
#plot_result_data(z, req_cor)

output_file = "./corrvec/corvecs_cor" + str(req_cor) + "_dim" + str(dim) + "_num" + str(n) + ".txt"
z_list = z.astype(np.float32).tolist()
save_vecs(z_list, output_file)
