#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

import tensorflow as tf

c = tf.constant('hello,server1')
# 生成一个有两个任务的集群，一个任务跑在本地2222端口，另外一个跑在2223端口
cluster = tf.train.ClusterSpec(
    {
        'local': ['localhost:2222', 'localhost:2223']
    }
)

# 通过上面生成的集群配置生成server,并job_name和tast_index指定当前所启动的任务，第一个任务task_index值为0
server = tf.train.Server(cluster, job_name='local', task_index=0)

# 通过server.target生成会话来使用tensorflow集群中资源，通过设置log_device_placement可以看到执行每一个操作的任务
sess = tf.Session(server.target,config=tf.ConfigProto(log_device_placement=True))

print(sess.run(c))
server.join()
