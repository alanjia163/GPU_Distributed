#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

import tensorflow as tf

c = tf.constant('hello,distributed tensorflow!')

# 创建一个本地Tensorflow 集群
server = tf.train.Server.create_local_server()

# 在集群上创建一个会话
sess = tf.Session(server.target)

print(sess.run(c))
