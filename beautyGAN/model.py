import tensorflow.compat.v1 as tf
import numpy as np
import os
import glob
from imageio import imread, imsave
import cv2

def preprocess(img):
    return (img / 255. - 0.5) * 2

def deprocess(img):
    return (img + 1) / 2

class beautyGAN:
    def __init__(self, img_size=256):
        tf.disable_eager_execution()
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph(os.path.join('beautyGAN/model', 'model.meta'))
        saver.restore(self.sess, tf.train.latest_checkpoint('beautyGAN/model'))
        graph = tf.get_default_graph()
        self.X = graph.get_tensor_by_name('X:0')
        self.Y = graph.get_tensor_by_name('Y:0')
        self.Xs = graph.get_tensor_by_name('generator/xs:0')
        self.img_size = img_size
        
    def generate(self, input_img, makeups_path, n=10):
        no_makeup = cv2.resize(imread(input_img), (self.img_size, self.img_size))
        X_img = np.expand_dims(preprocess(no_makeup), 0)
        makeups = glob.glob(os.path.join(makeups_path, '*.png'))
        makeups = np.random.choice(makeups, n, replace=False)
        result = np.ones((2 * self.img_size, (len(makeups) + 1) * self.img_size, 3))
        result[self.img_size: 2 * self.img_size, :self.img_size] = no_makeup / 255.
        for i in range(len(makeups)):
            makeup = cv2.resize(imread(makeups[i]), (self.img_size, self.img_size))
            Y_img = np.expand_dims(preprocess(makeup), 0)
            Xs_ = self.sess.run(self.Xs, feed_dict={self.X: X_img, self.Y: Y_img})
            Xs_ = deprocess(Xs_)
            result[:self.img_size, (i + 1) * self.img_size: (i + 2) * self.img_size] = makeup / 255.
            result[self.img_size: 2 * self.img_size, (i + 1) * self.img_size: (i + 2) * self.img_size] = Xs_[0]

        imsave('result.jpg', result)