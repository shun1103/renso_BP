# -*- coding: utf-8 -*-

import os
import subprocess
import datetime
import argparse
import time

def all_test():
    hiddens = ['50', '100', '120', '150', '180', '200', '220', '250']
    ptns = ['1', '10', '25', '50', '75', '100', '120', '150', '180', '200', '220', '250', '280', '300']
    cors = ['0.2', '0.4', '0.6', '0.8']


    cmd = ['python3', 'script_test.py', '-hd']


    for count in range(1, 6):
        for hdn in hiddens:
            cmd.append(hdn)
            cmd.append('-p')
            for ptn in ptns:
                cmd.append(ptn)
                cmd.append('-c')
                for c in cors:
                    cmd.append(c)
                    cmd.append('-tn')
                    cmd.append(str(count))

                    subprocess.call(cmd)

                    cmd.pop()
                    cmd.pop()
                    cmd.pop()

                cmd.pop()
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


if __name__ == '__main__':
    all_test()