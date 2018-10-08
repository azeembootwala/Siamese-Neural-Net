import tensorflow as tf
from scipy.special import factorial



class Contrastive(object):
    def __init__(self,batch_size=16):
        self.batch_size = batch_size
        pass

    def pairwise_distances(self,embedding, squared=False):
        """Args:embedding: tensorof shape (batch_size, 128)
        Returns
        pairwaise_distances: tensor of shape(batch_size,batch_size)

        """
        # Getting dot product between all embedding
        dot_product = tf.matmul(embedding,tf.transpose(embedding))

        #Taking the l2 norm of each embedding
        square_norm = tf.diag_part(dot_product)

        # we now compute pairwise distance ||a - b||² = || a ||² - 2 <a,b> + ||b||²
        distances = tf.expand_dims(square_norm,0) - 2.0 * dot_product  + tf.expand_dims(square_norm,1)

        # Removing negative distances if any
        distances = tf.maximum(distances,0.0)
        if not squared:
            # The gradient of sqrt is infinite when distances = 0.0 so we add an epslilon

            # Creating a mask to add small epsilon values
            mask = tf.to_float(tf.equal(distances,0.0))

            distances = distances + mask * 1e-16

            distances = tf.sqrt(distances)

            distances = distances * (1.0-mask)
        ###Converting to vector
        distance_vec = []
        iter = -1
        for i in range(0,self.batch_size):
            iter+=1
            for j in range(iter,self.batch_size):
                if i==j:
                    pass
                else:
                    distance_vec.append(distances[i,j])

        return tf.convert_to_tensor(distance_vec) # distances

    def get_binaray_labels(self,labels):
        """If the two images come from the same class then label is 1
        else 0"""
        indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]),tf.bool)
        indices_not_equal = tf.logical_not(indices_equal)
        labels_equal = tf.equal(tf.expand_dims(labels,0), tf.expand_dims(labels,1))

        binary_labels = tf.to_float(labels_equal)

        labels_vec = []
        iter = -1
        for i in range(0,self.batch_size):
            iter+=1
            for j in range(iter,self.batch_size):
                if i==j:
                    pass
                else:
                    labels_vec.append(binary_labels[i,j])

        return tf.stack(labels_vec,0)


    def pair_combos(self,embedding):
        num_combinations = int(factorial(self.batch_size, exact = True)/(factorial(self.batch_size-2)*factorial(2)))
        #anchor_left = tf.zeros([num_combinations,tf.shape(embedding)[1]])
        #anchor_right= tf.zeros([num_combinations,tf.shape(embedding)[1]])
        anchor_left = []
        anchor_right = []
        iter = 0
        for i in range(self.batch_size):

            for j in range(i+1,self.batch_size):
                if i == j:
                    pass
                else:
                    anchor_left.append(embedding[i])
                    anchor_right.append(embedding[j])
        anchor_left = tf.stack(anchor_left, axis = 0)
        anchor_right = tf.stack(anchor_right, axis = 0)

        return anchor_left , anchor_right


    def contrastive_loss(self,labels,anchor_left, anchor_right, margin,squared = False):
        # We need this function to output the contrastive loss for a given batch
        # using the formula L = (Y)*0.5 (D_w)² + (1-Y)*0.5 * max(0,margin-(D_w)²)
        distances = tf.sqrt(tf.reduce_sum(tf.square(anchor_left-anchor_right),1))

        one = tf.constant(1.0)
        """Tensorflow implementation of the cost
        cost1 = tf.reduce_mean(
        labels * tf.square(distances) +
        (one - labels) * tf.square(tf.maximum(margin-distances,0.0))
        )
        Below implementation gives the same result only you gotta change reduce_sum to reduce_mean"""

        first_part =  tf.reduce_sum(tf.multiply(labels,tf.square(distances)))
        max_part = tf.square(tf.maximum(0.0,(margin-distances)))
        second_part = tf.reduce_sum(tf.multiply((one-labels),max_part))
        cost = tf.add(first_part,second_part)
        return cost

    def contrastive_loss_v1(self, labels , anchor_left,anchor_right ,margin_lower , margin_upper):
        """Contrastive loss with dual margin"""

        distances = tf.sqrt(tf.reduce_sum(tf.square(anchor_left-anchor_right),1))

        one = tf.constant(1.0)
        max_part_first = tf.square(tf.maximum(0.0,(distances-margin_lower)))
        first_part =  tf.reduce_sum(tf.multiply(labels,max_part_first))
        max_part_second = tf.square(tf.maximum(0.0,(margin_upper-distances)))
        second_part = tf.reduce_sum(tf.multiply((one-labels),max_part_second))
        cost = tf.add(first_part,second_part)
        return cost
