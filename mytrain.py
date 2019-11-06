import os
import time

import numpy as np
import tensorflow as tf
from skimage import io, transform
from tensorflow.keras.utils import to_categorical

import loss
import unet

train_imagetransform_path = 'data/training/images_5/'
train_labeltransform_path = 'data/training/labels_5/'

def readimg(img_path):
    """ 读取单张图片，并简单预处理。
    """
    if os.path.exists(img_path) == False:
        print('File: %s doesnt exist.' % img_path)
        exit(-1)
    img = io.imread(img_path)
    return img

def loaddataset():
    """ 加载数据集
    """
    images_list = os.listdir(train_imagetransform_path)

    images = np.array([readimg(train_imagetransform_path + name) for name in images_list])
    labels = np.array([readimg(train_labeltransform_path + name) for name in images_list])
    
    return images, labels

if __name__ == '__main__':
    x_place = tf.placeholder(shape=[None, 352, 1216, 3], name="x", dtype=tf.float32)
    y_place = tf.placeholder(shape=[None, 352, 1216, 6], name="y", dtype=tf.float32)
    keep_prob = tf.placeholder(name="dropout_probability", dtype=tf.float32)
    
    # 网络
    logits = unet.create_u_net(x_place, keep_prob, layers=5, features_root=64)
    # 损失
    weights = np.array([1, 1, 1, 1, 1, 1])
    cost = loss.softmax_cross_entropy(labels=y_place,
                                      logits=logits, 
                                      n_classes=6,
                                      weights=weights)
    # 优化器
    lr = tf.Variable(initial_value=1e-5, name='lr', trainable=False, dtype=tf.float64)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True

    #sess = tf.Session(config=config)
    sess = tf.Session()
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    sess.run(init)

    RELOAD = False
    if RELOAD:
        saver.restore(sess, 'log/model.ckpt')
    
    # 数据集
    x_train, y_train = loaddataset()  
    y_train_onehot = to_categorical(y_train)

    tx = x_train[:180]
    ty = y_train_onehot[:180]

    vx = x_train[180:]
    vy = y_train_onehot[180:]

    with open('log.txt', 'a') as f:
        print('train_x shape:', tx.shape, '\ntrain_y shape:', ty.shape, file=f)
        print('val_x shape:', vx.shape, '\nval_y shape:', vy.shape, file=f)
    print('train_x shape:', tx.shape, '\ntrain_y shape:', ty.shape)
    print('val_x shape:', vx.shape, '\nval_y shape:', vy.shape)
    
    # 训练
    min_val_cost = 0
    no_improve = 0
    for epoch in range(300):
        train_cost = 0
        val_cost = 0
        total_batch = 180
        val_batch = 20

        start = time.time()
        for i in range(total_batch):    
            feed_dict = {x_place:tx[i:i+1], y_place:ty[i:i+1], keep_prob:0.5}
            _, loss = sess.run((optimizer, cost), feed_dict=feed_dict)
            train_cost += loss
        end = time.time()
        
        for i in range(val_batch):       
            feed_dict = {x_place:vx[i:i+1], y_place:vy[i:i+1], keep_prob:1}
            val_cost += sess.run(cost, feed_dict=feed_dict)

        with open('./log.txt', 'a') as f:
            print('Epoch:{:n}'.format(epoch+1),
                  'train_cost={:.9f}'.format(train_cost/total_batch),
                  'val_cost={:.9f}'.format(val_cost/val_batch),
                  'train_time={:.9f}s'.format(end-start), file=f)
        print('Epoch:{:n}'.format(epoch+1),
              'train_cost={:.9f}'.format(train_cost/total_batch),
              'val_cost={:.9f}'.format(val_cost/val_batch),
              'train_time={:.9f}s'.format(end-start))
        
        # 打乱训练集顺序。
        index = np.arange(180)
        np.random.shuffle(index)
        tx = tx[index]
        ty = ty[index]

        # 存储模型、加载模型、调整学习率等。
        if epoch == 0:
            min_val_cost = val_cost
            saver.save(sess, 'log/model.ckpt')
            with open('./log.txt', 'a') as f:
                print('Save Model\n', file=f)
            print('Save Model\n')
        else:
            if min_val_cost >= val_cost:
                saver.save(sess, 'log/model.ckpt')
                with open('./log.txt', 'a') as f:
                    print('Save Model\n', file=f)
                print('Save Model\n')
                min_val_cost = val_cost
                no_improve = 0
            else:
                no_improve += 1
                with open('./log.txt', 'a') as f:
                    print('No Improve, dont save model, no_improve={:n}\n'.format(no_improve), file=f)
                print('No Improve, dont save model, no_improve={:n}\n'.format(no_improve))
            
        if (epoch+1) % 10 == 0:
            # 过10个epoch调整一次学习率
            lr = lr * 0.9
            with open('./log.txt', 'a') as f:
                print('Change lr to {:.9f}\n'.format(lr.eval(session=sess)), file=f)
            print('Change lr to {:.9f}\n'.format(lr.eval(session=sess)))