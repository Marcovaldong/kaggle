# -*- coding: utf-8 -*-

from __future__ import print_function
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
def getMedian(data):
    data = sorted(data)
    size = len(data)
    if size % 2 == 0:
        return (data[size/2] + data[size/2-1]) / 2.0
    if size % 2 == 1:
        return data[size/2]

def readTrain():
    filepath = './train.csv'
    data = pd.read_csv(filepath).values
    num = len(data)
    sex = {'male': 0, 'female': 1}
    embarked = {'C': 0, 'Q': 1, 'S': 2}
    for i in xrange(0, num):
        data[i][4] = sex[data[i][4]]
        print(i)
        try:
            data[i][11] = embarked[data[i][11]]
        except:
            data[i][11] = 3
    df = pd.DataFrame(data, columns=['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'])
    df.to_csv('./train1.csv')


def wash1():
    '''
    1) 将名字中的称谓改成了数字，将性别改成数字
    2) 将上传港口转换成数字，并将空缺补上
    3) 将姓名拆开，
    :return:
    '''
    Title = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Dr': 0, 'Don': 0, 'Rev': 0, 'Mme': 0,
             'Ms': 0, 'Major': 0, 'Lady': 0, 'Sir': 0, 'Mlle': 0, 'Col': 0, 'Capt': 0, 'the Countess': 0,
             'Jonkheer': 0}
    embarked = {'C': 1, 'S': 2, 'Q': 3}
    sex = {'male': 1, 'female': 2}
    filepath = './train.csv'
    data = pd.read_csv(filepath).values
    print(type(data))
    newData = []
    for i in xrange(0, len(data)):
        # print("{}'th data".format(i))
        one = []
        data[i][4] = sex[data[i][4]]
        if data[i][11] not in ['C', 'S', 'Q']:
            data[i][11] = 1
        else:
            data[i][11] = embarked[data[i][11]]
        one.append(data[i][0])
        one.append(data[i][1])
        one.append(data[i][2])
        # print data[i][3]
        tmp1, tmp = data[i][3].split(', ')
        tmp2 = tmp.split('. ')[0]
        tmp3 = tmp.replace(tmp2 + '. ', '')
        one.append(tmp1)
        one.append(Title[tmp2])
        one.append(tmp3)
        for feature in data[i][4:]:
            one.append(feature)
        newData.append(one)

    df = pd.DataFrame(newData,
                      columns=['PassengerId', 'Survived', 'Pclass', 'Name', 'Title', 'Xing', 'Sex', 'Age', 'SibSp',
                               'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'])
    df.to_csv('./washed1.csv')

def wash2():
    '''
    船票ID分两种，一种是数字，另一种是字符串，这里仅仅将二者区分开来，没有做进一步分析
    :return:
    '''
    filepath = './washed1.csv'
    data = pd.read_csv(filepath).values
    num = 0
    # print(data[1][12])
    # print(type(data[1][12]))

    for i in xrange(len(data)):
        if type(data[i][12]) == str:
            num += 1
            data[i][12] = 1
        else:
            data[i][12] = 2
            continue
    df = pd.DataFrame(data, columns=['PassengerId', 'Survived', 'Pclass', 'Name', 'Title', 'Xing', 'Sex', 'Age', 'SibSp',
                               'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'])
    df.to_csv('./washed2.csv')

def wash3():
    '''
    将年龄空缺补上
    :return:
    '''
    filepath = './washed2.csv'
    data = pd.read_csv(filepath).values
    class1_male = []
    class1_female = []
    class2_male = []
    class2_female = []
    class3_male = []
    class3_female = []
    for i in xrange(len(data)):
        if data[i][2] == 1 and data[i][6] == 1 and not math.isnan(data[i][7]):
            class1_male.append(data[i][7])
        elif data[i][2] == 1 and data[i][6] == 2 and not math.isnan(data[i][7]):
            class1_female.append(data[i][7])
        elif data[i][2] == 2 and data[i][6] == 1 and not math.isnan(data[i][7]):
            class2_male.append(data[i][7])
        elif data[i][2] == 2 and data[i][6] == 2 and not math.isnan(data[i][7]):
            class2_female.append(data[i][7])
        elif data[i][2] == 3 and data[i][6] == 1 and not math.isnan(data[i][7]):
            class3_male.append(data[i][7])
        elif not math.isnan(data[i][7]):
            class3_female.append(data[i][7])

    for i in xrange(len(data)):
        if data[i][2] == 1 and data[i][6] == 1 and math.isnan(data[i][7]):
            data[i][7] = getMedian(class1_male)
        elif data[i][2] == 1 and data[i][6] == 2 and math.isnan(data[i][7]):
            data[i][7] = getMedian(class1_female)
        elif data[i][2] == 2 and data[i][6] == 1 and math.isnan(data[i][7]):
            data[i][7] = getMedian(class2_male)
        elif data[i][2] == 2 and data[i][6] == 2 and math.isnan(data[i][7]):
            data[i][7] = getMedian(class2_female)
        elif data[i][2] == 3 and data[i][6] == 1 and math.isnan(data[i][7]):
            data[i][7] = getMedian(class3_male)
        elif math.isnan(data[i][7]):
            data[i][7] = getMedian(class3_female)

    df = pd.DataFrame(data, columns=['PassengerId', 'Survived', 'Pclass', 'Name', 'Title',
                                     'Xing', 'Sex', 'Age', 'SibSp','Parch',
                                     'Ticket', 'Fare', 'Cabin', 'Embarked'])
    df.to_csv('./washed3.csv')

def washTest1():
    '''首先观察发现，测试集上没有票价和登船港口这两个参数的缺失，所以这里首先将登船港口和性别转换成数字'''
    filepath = './test.csv'
    data = pd.read_csv(filepath).values
    embarked = {'C': 1, 'S': 2, 'Q': 3}
    sex = {'male': 1, 'female': 2}
    print(data.shape)
    for i in xrange(len(data)):
        data[i][3] = sex[data[i][3]]
        data[i][10] = embarked[data[i][10]]

    '''拆分名字，我们要的是中间的Title，其实这个数据参考意义可能不大'''
    '''整理ticket的ID，分成数字和字符串两种'''
    Title = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Dr': 0, 'Don': 0, 'Rev': 0, 'Mme': 0,
             'Ms': 0, 'Major': 0, 'Lady': 0, 'Sir': 0, 'Mlle': 0, 'Col': 0, 'Capt': 0, 'the Countess': 0,
             'Jonkheer': 0, 'Dona': 0}
    newData = []
    for i in xrange(len(data)):
        oneData = []
        # oneData.append(data[i][0])
        oneData.append(data[i][1])
        tmp1, tmp = data[i][2].split(', ')
        tmp2 = tmp.split('. ')[0]
        oneData.append(Title[tmp2])
        for j in range(3, 7):
            oneData.append(data[i][j])
        if type(data[i][7]) == str:
            oneData.append(1)
        else:
            oneData.append(2)
        oneData.append(data[i][8])
        oneData.append(data[i][10])
        newData.append(oneData)

    '''保存数据'''
    df = pd.DataFrame(newData, columns=['Pclass', 'Title', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked'])
    df.to_csv('test1.csv')

def washTest2():
    '''补全年龄，这里需要统计包括训练集和测试集在内的所有数据之后再进行补全'''
    trainData = pd.read_csv('./washed3.csv').values
    testData = pd.read_csv('./test1.csv').values
    class1_male = []
    class1_female = []
    class2_male = []
    class2_female = []
    class3_male = []
    class3_female = []
    for i in xrange(len(trainData)):
        if trainData[i][2] == 1 and trainData[i][4] == 1 and not math.isnan(trainData[i][5]):
            class1_male.append(trainData[i][5])
        elif trainData[i][2] == 1 and trainData[i][4] == 2 and not math.isnan(trainData[i][5]):
            class1_female.append(trainData[i][5])
        elif trainData[i][2] == 2 and trainData[i][4] == 1 and not math.isnan(trainData[i][5]):
            class2_male.append(trainData[i][5])
        elif trainData[i][2] == 2 and trainData[i][4] == 2 and not math.isnan(trainData[i][5]):
            class2_female.append(trainData[i][5])
        elif trainData[i][2] == 3 and trainData[i][4] == 1 and not math.isnan(trainData[i][5]):
            class3_male.append(trainData[i][5])
        elif trainData[i][2] == 3 and trainData[i][4] == 2 and not math.isnan(trainData[i][5]):
            class3_female.append(trainData[i][5])

    for i in xrange(len(testData)):
        if testData[i][0] == 1 and testData[i][2] == 1 and not math.isnan(testData[i][3]):
            class1_male.append(testData[i][3])
        elif testData[i][0] == 1 and testData[i][2] == 2 and not math.isnan(testData[i][3]):
            class1_female.append(testData[i][3])
        elif testData[i][0] == 2 and testData[i][2] == 1 and not math.isnan(testData[i][3]):
            class2_male.append(testData[i][3])
        elif testData[i][0] == 2 and testData[i][2] == 2 and not math.isnan(testData[i][3]):
            class2_female.append(testData[i][3])
        elif testData[i][0] == 3 and testData[i][2] == 1 and not math.isnan(testData[i][3]):
            class3_male.append(testData[i][3])
        elif testData[i][0] == 3 and testData[i][2] == 2 and not math.isnan(testData[i][3]):
            class3_female.append(testData[i][3])

    print(class3_male)
    '''现在补全年龄'''
    for i in xrange(len(testData)):
        if testData[i][0] == 1 and testData[i][2] == 1 and math.isnan(testData[i][3]):
            testData[i][3] = getMedian(class1_male)
        elif testData[i][0] == 1 and testData[i][2] == 2 and math.isnan(testData[i][3]):
            testData[i][3] = getMedian(class1_female)
        elif testData[i][0] == 2 and testData[i][2] == 1 and math.isnan(testData[i][3]):
            testData[i][3] = getMedian(class2_male)
        elif testData[i][0] == 2 and testData[i][2] == 2 and math.isnan(testData[i][3]):
            testData[i][3] = getMedian(class2_female)
        elif testData[i][0] == 3 and testData[i][2] == 1 and math.isnan(testData[i][3]):
            testData[i][3] = getMedian(class3_male)
        elif testData[i][0] == 3 and testData[i][2] == 2 and math.isnan(testData[i][3]):
            testData[i][3] = getMedian(class3_female)

    '''保存数据'''
    df = pd.DataFrame(testData,
                      columns=['Pclass', 'Title', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked'])
    df.to_csv('test2.csv')

def display1():
    filepath = './washed1.csv'
    data = pd.read_csv(filepath)
    print(data.shape)

    fig = plt.figure()
    fig.set(alpha=0.2)
    plt.subplot2grid((2, 3), (0, 0))
    data.Survived.value_counts().plot(kind='bar')
    plt.title(u'num of survived people (1 symbols survived)')
    plt.ylabel(u"num")

    plt.subplot2grid((2, 3), (0, 1))
    data.Pclass.value_counts().plot(kind='bar')
    plt.ylabel(u"num")
    plt.title(u"class of passengers")

    plt.subplot2grid((2, 3), (0, 2))
    plt.scatter(data.Survived, data.Age)
    plt.ylabel('age')
    plt.title(u"distribution according to age")

    plt.subplot2grid((2, 3), (1, 0), colspan=2)
    data.Age[data.Pclass==1].plot(kind='kde')
    data.Age[data.Pclass==2].plot(kind='kde')
    data.Age[data.Pclass==3].plot(kind='kde')
    plt.xlabel(u'age')
    plt.ylabel(u'density')
    plt.title(u"different cabins' age distribution")
    plt.legend((u'first', u'second', u'third'), loc='best')

    plt.subplot2grid((2, 3), (1, 2))
    data.Embarked.value_counts().plot(kind='bar')
    plt.title(u"Port of embark")
    plt.ylabel(u"num")

    plt.show()

def display2():
    filepath = './washed1.csv'
    data = pd.read_csv(filepath)
    fig = plt.figure()
    fig.set(alpha=0.3)
    Survived_0 = data.Pclass[data.Survived==0].value_counts()
    Survived_1 = data.Pclass[data.Survived==1].value_counts()
    df = pd.DataFrame({u"Survived": Survived_1, u"Unsurvived": Survived_0})
    df.plot(kind='bar', stacked=True)
    plt.title(u"distribution by passengers' class")
    plt.xlabel(u"passengers' class")
    plt.ylabel(u"nums")
    plt.show()

def display3():
    filepath = './washed1.csv'
    data = pd.read_csv(filepath)
    fig = plt.figure()
    fig.set(alpha=0.2)
    Survived_0 = data.Title[data.Survived==0].value_counts()
    Survived_1 = data.Title[data.Survived==1].value_counts()
    df = pd.DataFrame({u"Survived": Survived_1, u"Unsurvived": Survived_0})
    df.plot(kind='bar', stacked=True)
    plt.show()

def display4():
    filepath = './washed2.csv'
    data =  pd.read_csv(filepath)
    fig = plt.figure()
    fig.set(alpha=0.2)
    Survived_0 = data.Cabin[data.Survived==0].value_counts()
    Survived_1 = data.Cabin[data.Survived==1].value_counts()
    df = pd.DataFrame({u"Survived": Survived_1, u"Unsurvived": Survived_0})
    df.plot(kind='bar', stacked=True)
    plt.show()

def shuffle():
    filepath = './washed3.csv'
    data = pd.read_csv(filepath).values
    np.random.shuffle(data)
    label = data[:, 1]
    Y = []
    for i in xrange(len(label)):
        if label[i] == 1:
            Y.append([0.0, 1.0])
        else:
            Y.append([1.0, 0.0])
    # print np.shape(Y)
    Y = np.float32(Y)
    # print type(Y)
    X = data[:, 2:]
    # print np.shape(X)
    return X, Y


def train(batch_size=100):
    # filepath = './washed3.csv'
    # data = pd.read_csv(filepath).values
    # label = data[:, 1]
    # Y = []
    # for i in xrange(len(label)):
    #     if label[i] == 1:
    #         Y.append([0.0, 1.0])
    #     else:
    #         Y.append([1.0, 0.0])
    # print np.shape(Y)
    # Y =np.float32(Y)
    # print type(Y)
    # X = data[:, 2:]
    # print np.shape(X)
    W = tf.Variable(tf.truncated_normal([9, 2], stddev=0.1))
    b = tf.Variable(tf.zeros([2]))
    x = tf.placeholder(tf.float32, [None, 9])
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 2], name='label')
    # cross_entropy = - tf.reduce_sum(y_ * tf.log(y))
    cross_entropy = - tf.reduce_sum(y_*tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.003).minimize(cross_entropy)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        epochs = 100
        epoch = 1
        best = 0
        while epoch <= epochs:
            print("epoch:", epoch)
            X, Y = shuffle()
            for i in xrange(0, 890, batch_size):
                sess.run(train_step, feed_dict={x: X[i:i+batch_size], y_: Y[i:i+batch_size]})
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print("mini_batch", i, "~", i+batch_size, "of", epoch, "epochs")
                print("accuracy:", sess.run(accuracy, feed_dict={x: X, y_: Y}))
                best = max(best, sess.run(accuracy, feed_dict={x: X, y_: Y}))
            epoch += 1
    print("The best accuracy: ", best)

def NNtrain(batch_size=50, epochs=100):
    '''
    We construct a three layers model: input layer, hidden layer and output layer
    :param batch_size:
    :return:
    '''

    testData = pd.read_csv('./test2.csv')

    x = tf.placeholder(tf.float32, [None, 9], name="data")
    y_ = tf.placeholder(tf.float32, [None, 2], name='label')

    with tf.name_scope("layer_in"):
        W1 = tf.Variable(tf.truncated_normal([9, 256],stddev=0.1), name="W1")
        b1 = tf.Variable(tf.truncated_normal([256], stddev=0.1), name="b1")
        hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1, name="hidden1")
    with tf.name_scope("layer_hidden_1"):
        W2 = tf.Variable(tf.truncated_normal([256, 128], stddev=0.1), name="W2")
        b2 = tf.Variable(tf.truncated_normal([128], stddev=0.1), name="b2")
        hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2, name="hidden2")
    with tf.name_scope("layer_hidden_2"):
        W3 = tf.Variable(tf.truncated_normal([128, 64]), name="W3")
        b3 = tf.Variable(tf.truncated_normal([64]), name="b3")
        hidden3 = tf.nn.relu(tf.matmul(hidden2, W3) + b3)

    with tf.name_scope("layer_out"):
        W4 = tf.Variable(tf.truncated_normal([64, 2], stddev=0.1), name="W4")
        b4 = tf.Variable(tf.truncated_normal([2], stddev=0.1), name="b4")
        y = tf.nn.softmax(tf.matmul(hidden3, W4) + b4, name="y_out")

    with tf.name_scope("cost"):
        vars = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'b' not in v.name]) * 0.005
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)+lossL2)
        # cross_entropy = - tf.reduce_sum(y_ * tf.log(y + 1e-10))
    with tf.name_scope("train"):
        train_step = tf.train.GradientDescentOptimizer(learning_rate=0.003).minimize(cost)
    with tf.name_scope("predict"):
        predict_step = tf.argmax(y, 1)

    with tf.name_scope("save_params"):
        saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        epoch = 1
        best = 0
        while epoch <= epochs:
            print("epoch: ", epoch)
            X, Y = shuffle()
            for i in xrange(0, 890, batch_size):
                sess.run(train_step, feed_dict={x: X[i:i + batch_size], y_: Y[i:i + batch_size]})
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print("mini_batch", i, "~", i + batch_size, "of", epoch, "epochs")
                print("accuracy: {}, the current best accuracy: {}".format(sess.run(accuracy, feed_dict={x: X, y_: Y}), best))
                # best = max(best, sess.run(accuracy, feed_dict={x: X, y_: Y}))
                if best < sess.run(accuracy, feed_dict={x: X, y_: Y}):
                    best = sess.run(accuracy, feed_dict={x: X, y_: Y})
                    saver.save(sess, "./save.ckpt")
                    savePredict(sess.run(predict_step, feed_dict={x: testData}))
            epoch += 1
        print("The best accuracy: ", best)


def savePredict(predict):
    data = []
    for i in xrange(418):
        one = []
        one.append(892+i)
        one.append(predict[i])
        data.append(one)
    df = pd.DataFrame(data, columns={"PassengerId", "Survived"})
    df.to_csv('./predict1.csv')

if __name__ == '__main__':
    start = time.time()
    # train()
    NNtrain(batch_size=50, epochs=1000)
    end = time.time()
    print("time consumption is {}s".format(end - start))

 #    aStr = '0 0 0 0 0 0 0 1 0 0 0 0 1 0 1 1 0 0 0 0 1 0 0 1 1 0 1 0 0 0 0 1 0 1 1 0 0\
 # 0 0 1 0 0 0 0 1 0 0 0 1 0 1 0 1 1 0 1 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0\
 # 1 1 0 0 0 0 1 1 0 0 0 0 0 0 0 1 0 0 1 0 1 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0\
 # 0 0 0 1 0 0 1 1 0 1 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 1 1 1 1 0 0 0 1 0\
 # 0 0 1 0 1 0 1 0 1 0 0 0 0 1 0 0 0 0 1 1 0 0 0 0 0 0 0 1 1 1 1 1 0 1 1 0 1\
 # 0 1 0 1 0 0 0 1 0 1 0 1 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 1 0 0 1 0 1 1 0 0 0\
 # 0 0 0 0 0 0 0 0 1 1 0 0 1 0 1 0 0 1 0 0 1 0 0 1 1 0 0 0 1 0 1 0 0 0 0 0 0\
 # 0 0 0 0 1 0 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 0 1 0 1 1 0 0 1 0 0 1 0 0 1 0 0\
 # 1 0 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 1 0 1 1 0 0 0 0 0\
 # 0 0 0 0 0 1 0 0 0 1 1 0 0 0 0 0 0 1 0 1 1 1 0 0 0 0 0 1 1 0 0 1 1 0 1 1 0\
 # 0 1 0 0 1 1 0 0 0 1 0 0 0 0 0 1 0 0 0 1 1 0 1 0 0 1 0 1 0 0 1 0 1 1 0 0 0\
 # 1 0 1 0 1 0 0 1 0 0 0'
 #    predict = []
 #    aList = aStr.split(' ')
 #    for i in xrange(418):
 #        one = []
 #        one.append(892 + i)
 #        one.append(int(aList[i]))
 #        predict.append(one)
 #
 #    df = pd.DataFrame(predict, columns={"PassengerId", "Survived"})
 #    df.to_csv('./predict.csv')

    # washTest2()





