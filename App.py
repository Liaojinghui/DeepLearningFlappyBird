#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/2 上午9:28
# @Author  : Aries
# @Site    : 
# @File    : App.py
# @Software: PyCharm Community Edition
# @license : Copyright(C), Vincent
# @Contact : vvvvincentvan@gmail.com
from deep_learning.deep_q_network import createNetwork, trainNetwork
import tensorflow as tf


def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)


def main():
    playGame()


if __name__ == "__main__":
    main()
