#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

'''
如果GPU无法,使用CPU
'''
import tensorflow as tf
a_cpu = tf.Variable(0,name="a_cpu")
with tf.device('/gpu:0'):
    a_gpu = tf.Variable(0, name="a_gpu")

sess = tf.Session(config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
sess.run(tf.initialize_all_variables())