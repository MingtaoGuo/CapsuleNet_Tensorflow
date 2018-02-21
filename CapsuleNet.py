import tensorflow as tf
import tensorflow.contrib as contrib
import numpy as np
import scipy.io as sci

class CapsuleNet_DynamicRouting():

    def __init__(self, batchsize, nums_outputs, vec_len,iter, data):
        self.batchsize = batchsize#Set the batchsize, this code is 50
        self.nums_outputs = nums_outputs#The number of outputs
        self.vec_len = vec_len#The length of one Vector in PrimaryCaps
        self.r = iter#The iteration number of Dynamic Routing
        #data is '.mat'
        self.traindata = data["traindata"]/255.0
        self.trainlabel = data["trainlabel"]
        self.testdata = data["testdata"]/255.0
        self.testlabel = data["testlabel"]
        self.X = tf.placeholder(dtype=tf.float32, shape=[batchsize, 784], name="X")
        self.Y = tf.placeholder(dtype=tf.float32, shape=[batchsize, 10], name="Lable")
        self.sess = tf.InteractiveSession()
        pass

    def squash(self, s_j):
        #The activation
        scale = tf.reduce_sum(tf.square(s_j), axis=2, keep_dims=True) / (1 + tf.reduce_sum(tf.square(s_j), axis=2, keep_dims=True))
        Unit_s_j = s_j / (tf.sqrt(tf.reduce_sum(tf.square(s_j), axis=2, keep_dims=True)) + 1e-10)
        return scale * Unit_s_j

    def ROUTING(self, u_hat):
        #The function of Dynamic Routing
        u_hat = tf.stop_gradient(u_hat)
        u_hat = tf.squeeze(u_hat, axis=-2)
        b_ij = tf.zeros([1152, 10])
        for r in range(self.r):
            c_ij = tf.nn.softmax(b_ij, dim=-1)#c_ij:[1152, 10]
            c_ij = tf.reshape(c_ij, [1, 1152, 10, 1])#c_ij:[50, 1152, 10, 1]
            c_ij = tf.tile(c_ij, [self.batchsize, 1, 1, 1])
            s_j = tf.reduce_sum(c_ij * u_hat, axis=1)#s_j:[50, 10, 16]
            s_j = tf.reshape(s_j, [self.batchsize, self.nums_outputs, 16, 1])#s_j:[50, 10, 16, 1]
            v_j = self.squash(s_j)#v_j:[50, 10, 16, 1]
            b_ij = b_ij + tf.squeeze(tf.matmul(tf.transpose(tf.tile(tf.reshape(v_j, [self.batchsize, 1, 10, 16, 1]), [1, 1152, 1, 1, 1]), [0, 1, 2, 4, 3]),
                                               tf.reshape(u_hat, [self.batchsize, 1152, 10, 16, 1])))
            b_ij = b_ij[0, :, :]
        return tf.squeeze(v_j)#v_j:[50, 10, 16]

    def CapsuleLayer(self, u_i):
        W = tf.get_variable(name="Capsule_Weight", shape=[1, np.size(u_i, 1), self.nums_outputs, 8, 16], dtype=tf.float32,
                            initializer=contrib.layers.xavier_initializer())
        W_tile = tf.tile(W, [self.batchsize, 1, 1, 1, 1])
        u_i = tf.reshape(u_i, [self.batchsize, 1152, 1, 8, 1])
        u_i = tf.tile(u_i, [1, 1, 10, 1, 1])#keep the dim same with the W
        u_i = tf.transpose(u_i, [0, 1, 2, 4, 3])
        u_hat = tf.matmul(u_i, W_tile)
        del W_tile
        v_j = self.ROUTING(u_hat)
        return v_j

    def Loss(self, v_k, m_plus=0.9, m_min=0.1, lambd=0.5, scale_rec=0.0005):
        abs_vk = tf.sqrt(tf.reduce_sum(tf.square(v_k), axis=-1))
        L_k = self.Y * tf.square(tf.maximum(0., m_plus - abs_vk)) + lambd * (1 - self.Y) * tf.square(tf.maximum(0., abs_vk - m_min))
        loss1 = tf.reduce_sum(L_k)
        loss2 = self.Reconstruct(v_k)
        return loss1 + loss2 * scale_rec


    def CapsuleNet(self):
        images = tf.reshape(self.X, shape=[self.batchsize, 28, 28, 1])
        with tf.variable_scope("Conv1"):
            conv1 = contrib.layers.conv2d(inputs=images, num_outputs=256, kernel_size=[9, 9],
                                          stride=1, weights_initializer=contrib.layers.xavier_initializer_conv2d(),
                                          weights_regularizer=contrib.layers.l2_regularizer,
                                          padding="VALID", activation_fn=tf.nn.relu)
        with tf.variable_scope("Conv2"):
            conv2 = contrib.layers.conv2d(inputs=conv1, num_outputs=256, kernel_size=[9, 9],
                                          stride=2, weights_initializer=contrib.layers.xavier_initializer_conv2d(),
                                          weights_regularizer=contrib.layers.l2_regularizer,
                                          padding="VALID", activation_fn=tf.nn.relu)
        with tf.variable_scope("PrimaryCaps"):
            self.primarycaps = tf.reshape(conv2, shape=[self.batchsize, 1152, 8, 1])
            u_i = self.squash(self.primarycaps)
        with tf.variable_scope("CapsuleLayer_1"):
            self.DigitCaps = self.CapsuleLayer(u_i)
            self.loss = self.Loss(self.DigitCaps)
            pass

    def Reconstruct(self, DigitCaps):
        target = tf.matmul(tf.reshape(self.Y, [self.batchsize, 1, 10]), DigitCaps)#Representation of the reconstruction target:[50, 1, 16]
        target = tf.squeeze(target)#[50, 16]
        fc1 = contrib.layers.fully_connected(inputs=target, num_outputs=512, activation_fn=tf.nn.relu)
        fc2 = contrib.layers.fully_connected(inputs=fc1, num_outputs=1024, activation_fn=tf.nn.relu)
        self.fc_sigmoid = contrib.layers.fully_connected(inputs=fc2, num_outputs=784, activation_fn=tf.nn.sigmoid)
        return tf.reduce_mean(tf.reduce_sum(tf.square(self.X - self.fc_sigmoid), axis=-1))

    def get_acc(self):
        prediction = tf.sqrt(tf.squeeze(tf.reduce_sum(tf.square(self.DigitCaps), axis=-1)))
        correct_prediction = tf.equal(tf.argmax(prediction, axis=-1), tf.argmax(self.Y, axis=-1))
        acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return acc


    def train(self):
        train_step = tf.train.AdamOptimizer(0.0001).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        k = 0
        for i in range(10000):
            batch0 = self.traindata[k:k + self.batchsize, :]
            batch1 = self.trainlabel[k:k + self.batchsize, :]
            k = k + self.batchsize
            if k >= np.size(self.traindata, 0):
                perm = np.arange(self.traindata, 0)
                np.random.shuffle(perm)
                self.traindata = self.traindata[perm]
                self.trainlabel = self.trainlabel[perm]
                k = 0
            self.sess.run(train_step, feed_dict={self.X: batch0, self.Y: batch1})
            if i % 1 == 0:
                Trainacc = self.sess.run(self.get_acc(), feed_dict={self.X: batch0, self.Y: batch1})
                print("Step %g,Train Accuracy:%g"%(i, Trainacc))
            if i % 1 == 0:
                Testacc = self.sess.run(self.get_acc(), feed_dict={self.X: self.testdata[0:50, :], self.Y: self.testlabel[0:50, :]})
                print("Test Accuracy:%g"%(Testacc))
        pass

if __name__ == "__main__":
    data = sci.loadmat("C://Users//gmt//Desktop//randmnist.mat")
    capsule = CapsuleNet_DynamicRouting(batchsize=50, nums_outputs=10, vec_len=8, iter=3, data=data)
    capsule.CapsuleNet()
    capsule.train()
