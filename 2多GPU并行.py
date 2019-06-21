#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

from datetime import datetime
import os
import time

import tensorflow as tf
import mnist_inference

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  # they has been normalized to range (0,1)

# Hyper parameters
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.001
REGULARAZTION_RATE = 0.0001
TRAING_STEPS = 1000
MOVING_AVERAGE_DECAY = 0.99
N_GPU = 2

# 定义日志和模型输出路径
MODEL_SAVE_PATH = "logs_and_models"
MODEL_NAME = "model.ckpt"

# 数据存储路径,以入队列方式从TFRecord中读入数据
DATA_PATH = "data.tfrecords"


# 定义输入队列得到训练数据
def get_input():
    filename_queue = tf.train.string_input_producer([DATA_PATH])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # 定义数据解析格式
    features = tf.parse_single_example(
        serialized_example,
        features={
            "image_raw": tf.FixedLenFeature([], tf.string),
            "pixels": tf.FixedLenFeature([], tf.int64),
            "labels": tf.FixedLenFeature([],tf.int64)
        }
    )
    #解析图片和标签
    decoded_image = tf.decode_raw(features["image_raw"],tf.uint8)
    reshape_image = tf.reshape(decoded_image,[784])
    retyped_image = tf.cast(reshaped_image,tf.float32)
    label = tf.cast(features["label"],tf.int32)

    #定义输入队列并返回
    min_after_dequeue = 10000
    capacity = min_after_dequeue+3*BATCH_SIZE
    return tf.train.shuffle_batch(
        [retyped_image,label],
        batch_size=BATCH_SIZE,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )

#定义损失函数,不同的GPU计算的正则化损失都加入名为loss集合
def get_loss(x,y_,regularizer,scope):
    y = mnist_inference.inference(x,regularizer)

    cross_entropy =tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y,y_))
    regularization_loss = tf.add_n(tf.get_collection("losses",scope))

    loss  = cross_entropy + regularizer_loss
    return loss


def average_gradients(tower_grads):
    average_grads=[]
    for grad_and_vars in zip(*tower_grads):
        