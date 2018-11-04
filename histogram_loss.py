import numpy as np
import tensorflow as tf
import torch


class histogram(object):
    def __init__(self, num_steps):
        self.num_steps = num_steps
        self.step = 2 / (self.num_steps-1)
        torch_t = (torch.range(-1, 1, self.step).view(-1, 1)).numpy()
        self.t = tf.convert_to_tensor(torch_t)
        #self.t = self.my_tf_round(tf.reshape(tf.range(-1 , 1+self.step, self.step),[-1 ,1]))
        #one = tf.constant(1.0,shape=[1,])
        #self.t = tf.reshape(tf.concat([self.my_tf_round(tf.range(-1 , 1, self.step)), one], axis = 0),[-1,1])

        self.tsize = tf.shape(self.t)[0]

    def my_tf_round(self,x, decimals = 4):
        multiplier = tf.constant(10**decimals, dtype=x.dtype)
        #return tf.round(x * multiplier) / multiplier
        return x

    def positive_mask(self,classes):
        indices_equal=tf.cast(tf.eye(tf.shape(classes)[0]),tf.bool)
        indices_not_equal = tf.logical_not(indices_equal)

        labels_equal = tf.equal(tf.expand_dims(classes,0),tf.expand_dims(classes, 1))

        mask = tf.logical_and(indices_not_equal, labels_equal)

        return mask

    def negative_mask(self, classes):
        labels_equal = tf.equal(tf.expand_dims(classes,0), tf.expand_dims(classes,1))
        mask = tf.logical_not(labels_equal)
        return mask

    def histogram(self, inds, size):
        similarity = tf.identity(self.similarity)
        indsa = tf.logical_and(tf.equal(self.delta_repeat , (self.t - self.step)),tf.cast(inds,tf.bool)) # cast to int or not
        indsb = tf.logical_and(tf.equal(self.delta_repeat, self.t),tf.cast(inds,tf.bool))
        #mask = 1.0 - tf.cast(tf.logical_not(tf.logical_or(indsa, indsb)),tf.float32)
        mask_indsa = 1.0 - tf.cast(tf.logical_not(indsa),tf.float32)
        mask_indsb = 1.0 -tf.cast(tf.logical_not(indsb), tf.float32)
        #similarity = self.similarity * G
        similarity_indsa = ((similarity - self.t + self.step)/self.step)*mask_indsa
        similarity_indsb = ((-similarity + self.t + self.step)/self.step) * mask_indsb
        res = similarity_indsa + similarity_indsb

        return tf.reduce_sum(res , axis = 1) / tf.cast(size, tf.float32)



    def hist_loss(self,embedding,classes):
        # features dims = (atch_size, 128)
        # classes dims = (batch_size)
        # Normalize the features (step 1)
        diag_mat = tf.logical_not(tf.cast(tf.eye(tf.shape(classes)[0]),tf.bool))
        mat = tf.matrix_band_part(tf.ones([tf.shape(classes)[0],tf.shape(classes)[0]]),0,-1)
        upper_mat = tf.logical_and(diag_mat,tf.cast(mat,tf.bool))
        #embedding = tf.nn.l2_normalize(features, dim = 1) # take this out in final run
        pos_inds = tf.expand_dims(tf.boolean_mask(tf.cast(self.positive_mask(classes),tf.int32),upper_mat),0)
        neg_inds = tf.expand_dims(tf.boolean_mask(tf.cast(self.negative_mask(classes),tf.int32),upper_mat),0)
        num_positives = tf.reduce_sum(pos_inds)
        num_negatives = tf.reduce_sum(neg_inds)

        pos_inds = tf.tile(pos_inds,[self.num_steps,1])
        neg_inds = tf.tile(neg_inds,[self.num_steps,1])

        distances = tf.matmul(embedding , tf.transpose(embedding))
        similarity = tf.reshape(tf.boolean_mask(distances,upper_mat),[1,-1])
        self.similarity = tf.tile(similarity,[self.num_steps,1])
        self.delta_repeat = self.my_tf_round((tf.floor((self.similarity + 1)/self.step) * self.step -1 ))
        hist_pos = self.histogram(pos_inds, num_positives)
        hist_neg = tf.reshape(self.histogram(neg_inds , num_negatives),[-1,1])
        hist_pos_repeat = tf.tile(tf.reshape(hist_pos,[-1,1]),[1,self.num_steps])
        diag_mat1 = tf.logical_not(tf.cast(tf.eye(tf.shape(hist_pos_repeat)[0]),tf.bool))
        mat1 = tf.matrix_band_part(tf.ones([tf.shape(hist_pos_repeat)[0],tf.shape(hist_pos_repeat)[0]]),-1,0)
        hist_pos_inds = tf.cast(tf.logical_not(tf.logical_and(diag_mat1,tf.cast(mat1,tf.bool))),tf.float32)
        hist_pos_repeat = hist_pos_repeat * hist_pos_inds
        hist_pos_cdf = tf.reshape(tf.reduce_sum(hist_pos_repeat,axis = 0),[1,-1])
        loss = tf.reduce_sum(tf.matmul(hist_pos_cdf, hist_neg))
        return loss




if __name__ == "__main__":
    # creating a dummy feature tensor
    np.random.seed(1123)
    num_steps = 150
    batch_size = 16
    feature = np.random.randn(batch_size,128).astype(np.float32)
    classes = np.random.randint(0,9, batch_size)
    print(classes)
    # creating place holders for tensorflow graph
    X = tf.placeholder(tf.float32,shape=(None, 128))
    Y = tf.placeholder(tf.int32, shape=(None,))
    hist_obj = histogram(num_steps)
    output = hist_obj.hist_loss(X, Y)
    with tf.Session() as sess:
        out = sess.run(output, feed_dict={X:feature , Y:classes})
        print(out)
        #np.savetxt("out_tensorflow.txt", out)
