# -*- coding: utf-8 -*-

import os
import subprocess
import datetime
import argparse
import time

def all_test():
    noise_rates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    parser = argparse.ArgumentParser(description='乱数ベクトルの想起')

    parser.add_argument('-hd', '--hdn', default=64, help='中間層の大きさ')
    parser.add_argument('-c', '--cor', help='データの相関係数')
    parser.add_argument('-p', '--ptn', help='記憶させるパターン数。最大1000')

    args = parser.parse_args()

    hidden_num = args.hdn
    cor = args.cor
    pattern_num = args.ptn


    cmd = ['python3', 'noise_test.py', '-hd', hidden_num, '-p', pattern_num, '-c', cor, '-nr']

    for nr in noise_rates:
        cmd.append(str(nr))
        subprocess.call(cmd)

        cmd.pop()


if __name__ == '__main__':
    all_test()