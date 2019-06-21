#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin
'''
第二个任务
'''
import tensorflow as tf
c = tf.constant('hello,server2')

#和第一个程序一样的集群配置，每个任务需要相同的配置
cluster = tf.train.ClusterSpec(
    {'local':['localhost:2222','localhost:2223']}
)
#指定task_index为1,这个程序将在localhost:2223启动
server = tf.train.Server(cluster,job_name='local',task_index=1)
# 通过server.target生成会话来使用tensorflow集群中资源，通过设置log_device_placement可以看到执行每一个操作的任务
sess = tf.Session(server.target,config=tf.ConfigProto(log_device_placement=True))

print(sess.run(c))
server.join()
