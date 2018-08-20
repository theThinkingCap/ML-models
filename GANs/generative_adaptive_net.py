import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pyPlot
import time

cos_ = lambda x: 100 + np.cos(x)
x_3 = lambda x: 100 + x**3

def sample_data(n=10000, scale=100, mapping = x_3):
    x = scale * (np.random.random_sample((n,1)) - 0.5)
    y = mapping(x)
    return np.concatenate((x,y), axis=1)

def sample_Z(m,n):
    return np.random.uniform(-1.,1.,size=[m,n])

class GenerativeAdversialNetV1:
    def __init__(self,featSize, genLearnR, discrimLearnR):
        GENERATOR = "Generator"
        DISCRIMINATOR = "Discriminator"
        self.X = tf.placeholder(tf.float32, [None, featSize])
        self.Z = tf.placeholder(tf.float32, [None, featSize])
        self.gen_sample = self.generator(self.Z,GENERATOR)
        self.r_logit = self.discriminator(self.X,DISCRIMINATOR)
        self.f_logit = self.discriminator(self.gen_sample,DISCRIMINATOR, reuse=True)

        self.discrim_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.r_logit,labels=tf.ones_like(self.r_logit))
                                           + tf.nn.sigmoid_cross_entropy_with_logits(logits=self.f_logit,labels=tf.zeros_like(self.f_logit)))
        self.gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.f_logit,labels=tf.ones_like(self.f_logit)))

        self.gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=GENERATOR)
        self.discrim_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=DISCRIMINATOR)

        self.gen_step = tf.train.RMSPropOptimizer(learning_rate=genLearnR).minimize(self.gen_loss,var_list=self.gen_vars)
        self.discrim_step = tf.train.RMSPropOptimizer(learning_rate=discrimLearnR).minimize(self.discrim_loss,var_list=self.discrim_vars)

    def custom_minimize(self):
        comp_grad = tf.train.RMSPropOptimizer(learning_rate=0.001).compute_gradients(self.gen_loss, var_list=self.gen_vars)
        for pairs in comp_grad:
            pairs[0] = -pairs[0]
        return tf.train.RMSPropOptimizer(learning_rate=0.001).apply_gradients(comp_grad)

    def generator(self,Z,var_scope,hLSize=[32,16,16],reuse=False):
        with tf.variable_scope(var_scope,reuse=reuse):
            h1 = tf.layers.dense(Z,hLSize[0],activation=tf.nn.leaky_relu)
            h2 = tf.layers.dense(h1,hLSize[1],activation=tf.nn.leaky_relu)
            h3 = tf.layers.dense(h2,hLSize[2],activation=tf.nn.leaky_relu)
            out = tf.layers.dense(h3,2)

        return out

    def discriminator(self,X,var_scope,hLsize=[32,16,16],reuse=False):
        with tf.variable_scope(var_scope,reuse=reuse):
            h1 = tf.layers.dense(X,hLsize[0],activation=tf.nn.leaky_relu)
            h2 = tf.layers.dense(h1,hLsize[1],activation=tf.nn.leaky_relu)
            h3 = tf.layers.dense(h2,hLsize[2],activation=tf.nn.leaky_relu)
            out = tf.layers.dense(h3,1)

        return out

    def generate_sample(self, sess, seed):
        return sess.run(self.gen_sample, feed_dict={self.Z: seed})

    def train(self, X_batch, Z_batch, sess, epoch=10001):
        tf.global_variables_initializer().run(session=sess)
        loss_track = []
        for i in range(epoch):
            #X_batch = sample_data(n=batch_size)
            #Z_batch = sample_Z(batch_size, 2)
            if i % 1000 == 0:
                gen_func = self.generate_sample(sess, Z_batch)
                print("coords " + str(gen_func[0,:]))

                pyPlot.plot(gen_func[:, 0], gen_func[:, 1], '.')
                pyPlot.show()
                pyPlot.gcf().clear()

            #for indd in range(40):
            _, dloss = sess.run([self.discrim_step, self.discrim_loss], feed_dict={self.X: X_batch, self.Z: Z_batch})
            _, gloss = sess.run([self.gen_step, self.gen_loss], feed_dict={self.Z: Z_batch})
            _ = sess.run(self.custom_minimize())
            print("Iteration: %d\t Discriminator Loss: %.4f\t Generator Loss: %.4f" % (i, dloss, gloss))

            if i % 1000 == 0:
                loss_track.append([dloss, gloss])
        return loss_track


if __name__ == '__main__':
    batch_size = 1024
    gan = GenerativeAdversialNetV1(2,0.001,0.001)
    X_train = sample_data(n=batch_size)
    Z_train = sample_Z(batch_size, 2)
    with tf.Session() as sess:
        loss_track = gan.train(X_train, Z_train, sess)

    loss_track = np.array(loss_track)

    pyPlot.figure()
    pyPlot.plot(loss_track[:,0])
    pyPlot.xlabel("Epoch")
    pyPlot.ylabel('Loss')
    pyPlot.title("Discriminator Loss")
    pyPlot.savefig('DiscriminatorLoss_.png')

    pyPlot.figure()
    pyPlot.plot(loss_track[:, 1])
    pyPlot.xlabel("Epoch")
    pyPlot.ylabel('Loss')
    pyPlot.title("Generator Loss")
    pyPlot.savefig("GeneratorLoss.png")
