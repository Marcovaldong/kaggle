# -*- coding: utf-8 -*-

from __future__ import print_function
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time

def featureChange1():
    filepath = './washed3.csv'
    data = pd.read_csv(filepath).values
    newData = []
    for i in xrange(len(data)):
        one = []
        one.append(data[i][0])
        one.append(data[i][1])
        if data[i][2] == 1:
            one.append(1)
            one.append(0)
            one.append(0)
        elif data[i][2] == 2:
            one.append(0)
            one.append(1)
            one.append(0)
        elif data[i][2] == 3:
            one.append(0)
            one.append(0)
            one.append(1)
        one.append(data[i][3])
        if data[i][4] == 1:
            one.append(1)
            one.append(0)
        else:
            one.append(0)
            one.append(1)
        one.append(data[i][5])
        one.append(data[i][6])
        one.append(data[i][7])
        one.append(data[i][8])
        if data[i][9] == 1:
            one.append(1)
            one.append(0)
        else:
            one.append(0)
            one.append(1)
        if data[i][10] == 1:
            one.append(1)
            one.append(0)
            one.append(0)
        elif data[i][10] == 2:
            one.append(0)
            one.append(1)
            one.append(0)
        else:
            one.append(0)
            one.append(0)
            one.append(1)
        newData.append(one)
    df = pd.DataFrame(newData, columns=['PassengerId', 'Survived', 'Pclass1', 'Pclass2', 'Pclass3',
                                        'Title', 'Male', 'Female', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin',
                                        'NotCabin', 'Embarked_C', 'Embarked_Q', 'Embarked_S'])
    df.to_csv('./trainFeature.csv')

def featureChange2():
    filepath = './test2.csv'
    data = pd.read_csv(filepath).values
    newData = []
    for i in xrange(len(data)):
        one = []
        if data[i][0] == 1:
            one.append(1)
            one.append(0)
            one.append(0)
        elif data[i][0] == 2:
            one.append(0)
            one.append(1)
            one.append(0)
        elif data[i][0] == 3:
            one.append(0)
            one.append(0)
            one.append(1)
        one.append(data[i][1])
        if data[i][2] == 1:
            one.append(1)
            one.append(0)
        else:
            one.append(0)
            one.append(1)
        one.append(data[i][3])
        one.append(data[i][4])
        one.append(data[i][5])
        one.append(data[i][6])
        if data[i][7] == 1:
            one.append(1)
            one.append(0)
        else:
            one.append(0)
            one.append(1)
        if data[i][8] == 1:
            one.append(1)
            one.append(0)
            one.append(0)
        elif data[i][8] == 2:
            one.append(0)
            one.append(1)
            one.append(0)
        else:
            one.append(0)
            one.append(0)
            one.append(1)
        newData.append(one)
    df = pd.DataFrame(newData, columns=['Pclass1', 'Pclass2', 'Pclass3',
                                        'Title', 'Male', 'Female', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin',
                                        'NotCabin', 'Embarked_C', 'Embarked_Q', 'Embarked_S'])
    df.to_csv('./testFeature.csv')

def pre():
    filepath = './trainFeature.csv'
    data = pd.read_csv(filepath).values
    print(np.shape(data))
    train_set = []
    validation_set = []
    for i in xrange(700):
        train_set.append(data[i])
    for i in xrange(700, len(data)):
        validation_set.append(data[i])
    # print(np.shape(train_set))
    # print(np.shape(validation_set))
    df1 = pd.DataFrame(train_set, columns=['PassengerId', 'Survived', 'Pclass1', 'Pclass2', 'Pclass3',
                                        'Title', 'Male', 'Female', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin',
                                        'NotCabin', 'Embarked_C', 'Embarked_Q', 'Embarked_S'])
    df1.to_csv('./train_set.csv')
    df2 = pd.DataFrame(validation_set, columns=['PassengerId', 'Survived', 'Pclass1', 'Pclass2', 'Pclass3',
                                        'Title', 'Male', 'Female', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin',
                                        'NotCabin', 'Embarked_C', 'Embarked_Q', 'Embarked_S'])
    df2.to_csv('./validation_set.csv')

def shuffle():
    filepath = './train_set.csv'
    data = pd.read_csv(filepath).values
    np.random.shuffle(data)
    label = data[:, 1]
    Y = []
    for i in xrange(len(label)):
        if label[i] == 1:
            Y.append([0.0, 1.0])
        else:
            Y.append([1.0, 0.0])
    label = np.float32(Y)
    X = data[:, 2:]
    print("Shuffling the training dataset")

    return X, Y

def getValidation():
    filepath = './validation_set.csv'
    data = pd.read_csv(filepath).values
    label = data[:, 1]
    Y = []
    for i in xrange(len(label)):
        if label[i] == 1:
            Y.append([0.0, 1.0])
        else:
            Y.append([1.0, 0.0])
    label = np.float32(Y)
    X = data[:, 2:]
    return X, Y

def savePredict(predict, filepath):
    data = []
    for i in xrange(418):
        one = []
        one.append(892+i)
        one.append(predict[i])
        data.append(one)
    df = pd.DataFrame(data, columns={"PassengerId", "Survived"})
    df.to_csv(filepath)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def NNtrain(batch_size=50, epochs=100, rate=0.003, filepath='./predict.csv'):
    '''
    We construct a three layers model: input layer, hidden layer and output layer
    :param batch_size:
    :return:
    '''

    testData = pd.read_csv('./testFeature.csv')

    x = tf.placeholder(tf.float32, [None, 15], name="data")
    y_ = tf.placeholder(tf.float32, [None, 2], name='label')

    with tf.name_scope("layer_in"):
        W_in = tf.Variable(tf.truncated_normal([15, 256],stddev=0.1), name="W1")
        b_in = tf.Variable(tf.truncated_normal([256], stddev=0.1), name="b1")
        hidden_in = tf.nn.relu(tf.matmul(x, W_in) + b_in, name="hidden1")
        hidden_in_flat = tf.reshape(hidden_in, [-1, 16, 16, 1], name="hidden1_flat")

    with tf.name_scope("conv_1"):
        W_conv1 = weight_variable([5, 5, 1, 16])
        b_conv1 = bias_variable([16])
        h_conv1 = tf.nn.relu(conv2d(hidden_in_flat, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope("conv_2"):
        W_conv2 = weight_variable([5, 5, 16, 32])
        b_conv2 = bias_variable([32])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope("layer_hidden_1"):
        W1 = weight_variable([4 * 4 * 32, 128])
        b1 = bias_variable([128])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 4 * 32])
        hidden1 = tf.nn.relu(tf.matmul(h_pool2_flat, W1) + b1)

    with tf.name_scope("layer_hidden_2"):
        W2 = weight_variable([128, 64])
        b2 = bias_variable([64])
        hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2)

    with tf.name_scope("layer_out"):
        W_out = weight_variable([64, 2])
        b_out = bias_variable([2])
        hidden_out = tf.nn.softmax(tf.matmul(hidden2, W_out) + b_out, name="output")
    # with tf.name_scope("layer_hidden_1"):
    #     W2 = tf.Variable(tf.truncated_normal([512, 128], stddev=0.1), name="W2")
    #     b2 = tf.Variable(tf.truncated_normal([128], stddev=0.1), name="b2")
    #     hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2, name="hidden2")
    # with tf.name_scope("layer_hidden_2"):
    #     W3 = tf.Variable(tf.truncated_normal([128, 64]), name="W3")
    #     b3 = tf.Variable(tf.truncated_normal([64]), name="b3")
    #     hidden3 = tf.nn.relu(tf.matmul(hidden2, W3) + b3, name="hidden3")
    #
    # with tf.name_scope("layer_out"):
    #     W4 = tf.Variable(tf.truncated_normal([64, 2], stddev=0.1), name="W4")
    #     b4 = tf.Variable(tf.truncated_normal([2], stddev=0.1), name="b4")
    #     y = tf.nn.softmax(tf.matmul(hidden3, W4) + b4, name="y_out")

    with tf.name_scope("cost"):
        vars = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'b' not in v.name]) * 0.05
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hidden_out, labels=y_)+lossL2)
        # cross_entropy = - tf.reduce_sum(y_ * tf.log(y + 1e-10))
    with tf.name_scope("train"):
        train_step = tf.train.GradientDescentOptimizer(learning_rate=rate).minimize(cost)
    with tf.name_scope("predict"):
        predict_step = tf.argmax(hidden_out, 1)

    with tf.name_scope("save_params"):
        saver = tf.train.Saver()
    # init = tf.global_variables_initializer()
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        epoch = 1
        best = 0
        while epoch <= epochs:
            print("epoch: ", epoch)
            train_X, train_Y = shuffle()
            validation_X, validation_Y = getValidation()
            for i in xrange(0, 700, batch_size):
                sess.run(train_step, feed_dict={x: train_X[i:i + batch_size], y_: train_Y[i:i + batch_size]})
                correct_prediction = tf.equal(tf.argmax(hidden_out, 1), tf.argmax(y_, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print("mini_batch", i, "~", i + batch_size, "of", epoch, "epochs")
                print("accuracy on train set: {}".format(sess.run(accuracy, feed_dict={x: train_X, y_: train_Y})))
                print("accuracy on validation set: {}, the current best accuracy: {}".format(sess.run(accuracy, feed_dict={x: validation_X, y_: validation_Y}), best))
                # best = max(best, sess.run(accuracy, feed_dict={x: X, y_: Y}))
                if best < sess.run(accuracy, feed_dict={x: validation_X, y_: validation_Y}):
                    best = sess.run(accuracy, feed_dict={x: validation_X, y_: validation_Y})
                    # saver.save(sess, "./save.ckpt")
                    savePredict(sess.run(predict_step, feed_dict={x: testData}), filepath=filepath)
            epoch += 1
        print("The best accuracy: ", best)


if __name__ == '__main__':
    # featureChange2()
    # pre()
    start = time.time()
    NNtrain(batch_size=50, epochs=300, rate=0.003, filepath='./predict35.csv')
    end = time.time()
    print("time consumption is {}s".format(end - start))

    # data = pd.read_csv('./predict15.csv').values
    # num = 0
    # for i in xrange(len(data)):
    #     if data[i][1] == 1:
    #         num += 1
    # print(num)
