import numpy as np
import tensorflow as tf


class Triplet(object):
    def __init__(self, batch_size):
        self.batch_size=batch_size

    def pairwise_distances(self, embedding, squared=False):
        """Args:embedding: tensor of shape (batch_size, 128)
                squared:   Boolean. if True output is pairwise
                           squared eucliden distance matrix
                           if False Output is pairwaise distance
                           Matrix
        Returns
        pairwaise_distances: tensor of shape (batch_size,batch_size)

        """
        dot_product = tf.matmul(embedding, tf.transpose(embedding))

        # take the l2 norm of each embedding
        square_norm = tf.diag_part(dot_product)

        # we now compute pairwise distance ||a - b||² = || a ||² - 2 <a,b> + ||b||²
        # This gives a symetric matrix with similar off diagonal elements
        distances = tf.expand_dims(square_norm,0) - 2.0 * dot_product  + tf.expand_dims(square_norm,1)

        # Some distances might be negative so we put zero there
        distances = tf.maximum(distances, 0.0)

        if not squared:
            # Because the gradient of root is infinite when distance = 0 eg on diagonal part
            # WE add small epsilon where distance is 0
            mask = tf.to_float(tf.equal(distances,0))
            distances = distances + mask * 1e-16
            distances = tf.sqrt(distances)

            # Correcting for epsilon
            distances = distances * (1.0 - mask)
        return distances

    def get_triplet_mask(self,labels):
        """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
        A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
        - i != j
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        """
        indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]),tf.bool)
        indices_not_equal = tf.logical_not(indices_equal)
        i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
        i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
        j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

        distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

        # Check if labels[i] == labels[j] and labels[i] != labels[k]
        labels_equal = tf.equal(tf.expand_dims(labels,0),tf.expand_dims(labels,1))
        i_equal_j = tf.expand_dims(labels_equal, 2)
        i_equal_k = tf.expand_dims(labels_equal, 1)

        valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

        # Combine the two masks
        mask = tf.logical_and(distinct_indices, valid_labels)

        return mask

    def get_anchor_positive_triplet_mask(self,labels):
        """Return a 2-D mask [a,p] is True iff a and p have same label and are distinct

            Args : Tensor with shape [batch_size]

        Returns:
            mask (2-D) Tensor with shape [batch_size, batch_size]
        """
        # Check if i and j are distinct
        indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
        indices_not_equal = tf.logical_not(indices_equal)

        # check if labels[i] == labels[j]
        labels_equal = tf.equal(tf.expand_dims(labels,0), tf.expand_dims(labels,1))

        # Combine the two masks
        mask = tf.logical_and(indices_not_equal,labels_equal)

        return mask

    def get_anchor_negative_triplet_mask(self, labels):

        labels_equal = tf.equal(tf.expand_dims(labels,0), tf.expand_dims(labels,1))
        mask = tf.logical_not(labels_equal)

        return mask


    def triplet_loss_batch_all(self,labels,embeddings,margin, squared=False):
        """Build the triplet loss over a batch of embeddings.

        We generate all the valid triplets and average the loss over the positive ones.

        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                            If false, output is the pairwise euclidean distance matrix.

            Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        # We first get the pairwise distance matrix
        pairwise_distance = self.pairwise_distances(embeddings, squared=squared)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)

        anchor_positive_distance = tf.expand_dims(pairwise_distance,2)
        anchor_negative_distance = tf.expand_dims(pairwise_distance,1)

        triplet_loss = anchor_positive_distance - anchor_negative_distance + margin

        # Put a zero to invalid triplets
        #(where label(a)!=label(p) or label(n)==label(a)or a==p)
        mask = self.get_triplet_mask(labels)
        mask = tf.to_float(mask)
        triplet_loss = tf.multiply(triplet_loss, mask)
        # Remove negative losses (i.e. the easy triplets)
        triplet_loss = tf.maximum(triplet_loss, 0.0)

        #count the number of valid triplets
        valid_triplets = tf.to_float(tf.greater(triplet_loss,1e-16))
        num_positive_triplets = tf.reduce_sum(valid_triplets)
        num_valid_triplets = tf.reduce_sum(mask)
        fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = tf.reduce_sum(triplet_loss)/ (num_positive_triplets + 1e-16)

        return  triplet_loss , fraction_positive_triplets

    def triplet_loss_batch_hard(self, labels , embedding, margin, squared=False):
        """Build the triplet loss over a batch of embeddings.

        For each anchor, we get the hardest positive and hardest negative to form a triplet.

        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                    If false, output is the pairwise euclidean distance matrix.

        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        # Get pairwaise distance matrix
        pairwise_distance = self.pairwise_distances(embedding, squared=squared)

        # Now for each anchor we want to get the hardest positive
        # First we need the mask for anchor-positive where all valid positives should have the same label
        mask_anchor_positive = self.get_anchor_positive_triplet_mask(labels)
        mask_anchor_positive = tf.to_float(mask_anchor_positive)

        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_distance = tf.multiply(mask_anchor_positive,pairwise_distance)

        hardest_positive_distance = tf.reduce_max(anchor_positive_distance,axis = 1,keepdims=True)

        # WE now get the hardest negative distance
        # First we need to get the mask for every valid negative (they should have different labels)
        mask_anchor_negative = self.get_anchor_negative_triplet_mask(labels)
        mask_anchor_negative = tf.to_float(mask_anchor_negative)

        max_anchor_negative_dist = tf.reduce_max(pairwise_distance, axis = 1, keepdims=True)
        anchor_negative_distance = pairwise_distance + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

        hardest_negative_distance = tf.reduce_min(anchor_negative_distance, axis = 1 , keepdims=True)

        triplet_loss = tf.maximum(hardest_positive_distance - hardest_negative_distance + margin , 0)

        triplet_loss = tf.reduce_mean(triplet_loss)

        return triplet_loss
