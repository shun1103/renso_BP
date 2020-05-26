# -*- coding: utf-8 -*-

import os
import subprocess
import datetime
import argparse
import time

def all_train():

    hiddens = '250'
    ptns = ['180', '200']
    cors = ['0.2', '0.4', '0.6', '0.8']
    #ptns = ['160', '170']


    cmd = ['python3', 'script_train.py', '-hd', hiddens, '-p']

    t_s = time.time()

    for ptn in ptns:
        cmd.append(ptn)
        cmd.append('-c')

        for c in cors:
            cmd.append(c)
            subprocess.call(cmd)

            cmd.pop()

        cmd.pop()
        cmd.pop()



    """
        for c in cor:
        cmd.append(c)
        cmd.append('-p')
        for ptn in ptns:
            cmd.append(ptn)
            subprocess.call(cmd)

            cmd.pop()
        cmd.pop()
        cmd.pop()
    """


    t_f = time.time()
    _train_time = t_f - t_s

    print('Train all!!')
    print('Train Time : ' + str(_train_time))


if __name__ == '__main__':
    all_train()