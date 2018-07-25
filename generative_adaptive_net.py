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

def generator(Z,hLSize=[16,16,16],reuse=False):
    with tf.variable_scope("Generator",reuse=reuse):
        h1 = tf.layers.dense(Z,hLSize[0],activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1,hLSize[1],activation=tf.nn.leaky_relu)
        h3 = tf.layers.dense(h2,hLSize[2],activation=tf.nn.leaky_relu)
        out = tf.layers.dense(h3,2)

    return out

def discriminator(X,hLsize=[16,16,16],reuse=False):
    with tf.variable_scope("Discriminator",reuse=reuse):
        h1 = tf.layers.dense(X,hLsize[0],activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1,hLsize[1],activation=tf.nn.leaky_relu)
        h3 = tf.layers.dense(h2,hLsize[2],activation=tf.nn.leaky_relu)
        h4 = tf.layers.dense(h3,2)
        out = tf.layers.dense(h4,1)

    return out

if __name__ == '__main__':
    # samples = sample_data(mapping=cos_)
    samples = sample_data()

    ## Set up mapping for Generator and Discriminator network ##
    X = tf.placeholder(tf.float32,[None,2])
    Z = tf.placeholder(tf.float32,[None,2])
    gen_sample = generator(Z)

    r_logit = discriminator(X)
    f_logit = discriminator(gen_sample,reuse=True)

    discrim_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logit,labels=tf.ones_like(r_logit)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit,labels=tf.zeros_like(f_logit)))
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit,labels=tf.ones_like(f_logit)))

    gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Generator")
    discrim_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Discriminator")

    gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss,var_list=gen_vars)
    discrim_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(discrim_loss,var_list=discrim_vars)

    batch_size = 256

    with tf.Session() as sess:
        tf.global_variables_initializer().run(session=sess)
        loss_track = []
        for i in range(10001):
            X_batch = sample_data(n=batch_size)
            Z_batch = sample_Z(batch_size, 2)
            if i % 1000 == 0:
                gen_func = sess.run(gen_sample,feed_dict={Z:Z_batch})
                pyPlot.plot(gen_func[:,0],gen_func[:,1],'.')
                pyPlot.show()
            for d_ind in range(10):
                _, dloss = sess.run([discrim_step,discrim_loss], feed_dict={X:X_batch, Z: Z_batch})
                _, gloss = sess.run([gen_step,gen_loss], feed_dict={Z:Z_batch})

            print("Iteration: %d\t Discriminator Loss: %.4f\t Generator Loss: %.4f"%(i,dloss,gloss))

            if i % 100 == 0:
                loss_track.append([dloss,gloss])

    loss_track = np.array(loss_track)

#    dFig,gFig = pyPlot.figure(1), pyPlot.figure(2)

    pyPlot.figure()
    pyPlot.plot(loss_track[:,0])
    pyPlot.xlabel("Epoch")
    pyPlot.ylabel('Loss')
    pyPlot.title("Discriminator Loss - Cos")
    pyPlot.savefig('DiscriminatorLossCosine.png')

    pyPlot.figure()
    pyPlot.plot(loss_track[:, 1])
    pyPlot.xlabel("Epoch")
    pyPlot.ylabel('Loss')
    pyPlot.title("Generator Loss - Cos")
    pyPlot.savefig("GeneratorLossCosine.png")
