import numpy as np
import os
import tensorflow as tf
from util_vgg import Normal , Xavier
from Generators import Generators
from glob import glob
import matplotlib.pyplot as plt
from inference import infer_generator
from evaluate import evaluate
from manifold_vizualization import tsne
np.set_printoptions(threshold=np.nan)

class Conv2Layer(object):
    def __init__(self,mi1, mo1,mo2,Initializer,fw=3, fh=3 , pool_sz =(2,2)):
        self.shape1 = (fw, fh, mi1, mo1)
        self.shape2 = (fw, fh, mo1, mo2)
        self.Initializer = Initializer
        w1, b1 = self.Initializer.init_filter(self.shape1)
        w2, b2 = self.Initializer.init_filter(self.shape2)
        self.w1 = tf.Variable(w1.astype(np.float32))
        self.b1 = tf.Variable(b1.astype(np.float32))
        self.w2 = tf.Variable(w2.astype(np.float32))
        self.b2 = tf.Variable(b2.astype(np.float32))
        self.params=[self.w1, self.b1, self.w2, self.b2]

    def forward(self, X):
        conv_out = tf.nn.conv2d(X,self.w1, strides=[1,1,1,1],padding="SAME", name="conv1-")#+str(iter))
        conv_out = tf.nn.bias_add(conv_out, self.b1, name ="bias1-")#+str(iter)
        conv_out = tf.nn.relu(conv_out)
        conv_out = tf.nn.conv2d(conv_out, self.w2, strides=[1,1,1,1], padding="SAME", name="conv2-")#+str(iter))
        conv_out = tf.nn.bias_add(conv_out, self.b2, name="bias2-")#str(iter)
        conv_out = tf.nn.relu(conv_out)
        pool_out = tf.nn.max_pool(conv_out,ksize=[1,2,2,1],strides=[1,2,2,1], padding="SAME")
        return pool_out
    def forward_without_pool(self, X, pad_type="VALID"):
            conv_out = tf.nn.conv2d(X,self.w1, strides=[1,1,1,1],padding=pad_type, name="conv1-")#+str(iter))
            conv_out = tf.nn.bias_add(conv_out, self.b1, name ="bias1-")#+str(iter)
            conv_out = tf.nn.relu(conv_out)
            return conv_out

class Conv3Layer(object):
    def __init__(self, mi1,mo1, mo2, mo3,Initializer,  fw=3, fh=3, pool_sz=(2,2)):
        self.shape1=(fw,fh,mi1,mo1)
        self.shape2=(fw,fh,mo1,mo2)
        self.shape3=(fw,fh,mo2,mo3)
        self.Initializer = Initializer
        w1, b1 = self.Initializer.init_filter(self.shape1)
        w2, b2 = self.Initializer.init_filter(self.shape2)
        w3, b3 = self.Initializer.init_filter(self.shape3)
        self.w1 = tf.Variable(w1.astype(np.float32))
        self.b1 = tf.Variable(b1.astype(np.float32))
        self.w2 = tf.Variable(w2.astype(np.float32))
        self.b2 = tf.Variable(b2.astype(np.float32))
        self.w3 = tf.Variable(w3.astype(np.float32))
        self.b3 = tf.Variable(b3.astype(np.float32))
        self.params = [self.w1,self.b1, self.w2, self.b2, self.w3, self.b3]

    def forward(self,X):
        conv_out = tf.nn.conv2d(X, self.w1,strides=[1,1,1,1], padding="SAME", name="conv1-")
        conv_out = tf.nn.bias_add(conv_out,self.b1, name ="bias1-")
        conv_out = tf.nn.relu(conv_out)
        conv_out = tf.nn.conv2d(conv_out, self.w2, strides=[1,1,1,1], padding="SAME", name ="conv2-")
        conv_out = tf.nn.bias_add(conv_out, self.b2, name ="bias2-")
        conv_out = tf.nn.relu(conv_out)
        conv_out = tf.nn.conv2d(conv_out, self.w3, strides=[1,1,1,1], padding="SAME", name ="conv3-")
        conv_out = tf.nn.bias_add(conv_out, self.b3 , name ="bias3-")
        conv_out = tf.nn.relu(conv_out)
        pool_out = tf.nn.max_pool(conv_out,ksize=[1,2,2,1],strides=[1,2,2,1], padding="SAME")
        return pool_out

class FCLayer(object):
    def __init__(self, M1, M2, Initializer):
        self.Initializer= Initializer
        self.M1 = M1
        self.M2 = M2
        W , b = self.Initializer.initialize_weights_bias(M1, M2)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        self.params = [self.W, self.b]

    def forward(self, X):
        out = tf.matmul(X, self.W)+self.b
        return out


class VGG(object):
    def __init__(self, conv2layer, conv3layer,FCLayer, Initializer, path, batch_size):
        self.conv2layer = conv2layer
        self.conv3layer = conv3layer
        self.FCLayer = FCLayer
        self.Initializer = Initializer
        self.conv_obj = []
        self.batch_size = batch_size
        self.val_iter = len(glob("../Data/dev-set/*/*.*")) #Number of iterations that we need to do to get an accuracy on val set complete
        self.path = path
        self.lr = 1e-5     #0.00001
        self.overall_MRR = []
        self.overall_MAP = []
        self.MAP_healthy =[]
        self.MAP_mild = []
        self.MAP_moderate =[]
        self.MAP_severe = []
        self.MAP_proliferative =[]
        self.MRR_healthy =[]
        self.MRR_mild = []
        self.MRR_moderate =[]
        self.MRR_severe = []
        self.MRR_proliferative =[]
        self.MAP_non_healthy =[]
        self.MRR_non_healthy =[]

    def plot_MAP(self,overall,healthy, mild, moderate,severe,proliferative,non_healthy,folder):
        print("Plotting MAP")
        fig1 = plt.figure()
        plt.plot(healthy,"-x",label="Healthy")
        plt.plot(mild,"-x",label="Mild")
        plt.plot(moderate,"-x",label="Moderate")
        plt.plot(severe,"-x",label="Severe")
        plt.plot(proliferative,"-x",label="Proliferative")
        plt.title("MAP after epoch: "+str(folder))
        plt.xlabel("Epochs")
        plt.ylabel("Mean Average Precision")
        plt.legend()
        fig1.savefig(os.path.join(self.path+"/"+str(folder),"Plots")+"/MAP-per-class_"+str(folder), transparent=False,bbox_inches = "tight" ,pad_inches=0)
        fig2 = plt.figure()
        plt.plot(overall,"-x",label="MAP_overall")
        plt.xlabel("Epochs")
        plt.ylabel("Mean Average Precision")
        plt.legend()
        fig2.savefig(os.path.join(self.path+"/"+str(folder),"Plots")+"/MAP-overall_"+str(folder), transparent=False,bbox_inches = "tight" ,pad_inches=0)
        fig3 = plt.figure()
        plt.plot(non_healthy,"-x",label="MAP_non_healthy")
        plt.xlabel("Epochs")
        plt.ylabel("Mean Average Precision")
        plt.legend()
        fig3.savefig(os.path.join(self.path+"/"+str(folder),"Plots")+"/MAP-non_healthy_"+str(folder), transparent=False,bbox_inches = "tight" ,pad_inches=0)

    def plot_MRR(self,overall,healthy, mild, moderate,severe,proliferative,non_healthy,folder):
        print("Plotting MRR")
        fig4 = plt.figure()
        plt.plot(healthy,"-x",label="Healthy")
        plt.plot(mild,"-x",label="Mild")
        plt.plot(moderate,"-x",label="Moderate")
        plt.plot(severe,"-x",label="Severe")
        plt.plot(proliferative,"-x",label="Proliferative")
        plt.title("MRR after Epoch: "+str(folder))
        plt.xlabel("Epochs")
        plt.ylabel("Mean Reciprocal Rank")
        plt.legend()
        fig4.savefig(os.path.join(self.path+"/"+str(folder),"Plots")+"/MRR-per-class_"+str(folder), transparent=False,bbox_inches = "tight" ,pad_inches=0)
        fig5 = plt.figure()
        plt.plot(overall,"-x",label="MRR_overall"+str(folder))
        plt.xlabel("Epochs")
        plt.ylabel("Mean Reciprocal Rank")
        plt.legend()
        fig5.savefig(os.path.join(self.path+"/"+str(folder),"Plots")+"/MRR-overall_"+str(folder), transparent=False,bbox_inches = "tight" ,pad_inches=0)
        fig6 = plt.figure()
        plt.plot(non_healthy,"-x",label="MRR_non_healthy"+str(folder))
        plt.xlabel("Epochs")
        plt.ylabel("Mean Reciprocal Rank")
        plt.legend()
        fig6.savefig(os.path.join(self.path+"/"+str(folder),"Plots")+"/MRR-non_healthy_"+str(folder), transparent=False,bbox_inches = "tight" ,pad_inches=0)

    def make_plots(self,session,folder, lr ):
        infer_generator(session,self.path+"/"+str(folder)+"/"+str(folder))
        E = evaluate(self.path+"/"+str(folder), 10)
        MRR = E.MRR()
        MAP, _ = E.MAP()
        self.overall_MRR.append(MRR)
        self.overall_MAP.append(MAP)
        MAP_per_class , MRR_per_class, _ = E.per_class_stats()
        self.MAP_healthy.append(MAP_per_class[0]) , self.MAP_mild.append(MAP_per_class[1]) , self.MAP_moderate.append(MAP_per_class[2])
        self.MAP_severe.append(MAP_per_class[3]) , self.MAP_proliferative.append(MAP_per_class[4])
        self.MRR_healthy.append(MRR_per_class[0]) , self.MRR_mild.append(MRR_per_class[1]) , self.MRR_moderate.append(MRR_per_class[2])
        self.MRR_severe.append(MRR_per_class[3]) , self.MRR_proliferative.append(MRR_per_class[4])
        MAP_non_healthy_val, _ = E.MAP_non_healthy()
        MRR_non_healthy_val = E.MRR_non_healthy()
        self.MAP_non_healthy.append(MAP_non_healthy_val)
        self.MRR_non_healthy.append(MRR_non_healthy_val)
        if not os.path.exists(os.path.join(self.path+"/"+str(folder),"Plots")):
            os.makedirs(os.path.join(self.path+"/"+str(folder),"Plots"))
        #plotting
        self.plot_MAP(self.overall_MAP,self.MAP_healthy,self.MAP_mild,self.MAP_moderate,self.MAP_severe,self.MAP_proliferative,self.MAP_non_healthy,folder)
        self.plot_MRR(self.overall_MRR,self.MRR_healthy,self.MRR_mild,self.MRR_moderate,self.MRR_severe,self.MRR_proliferative,self.MRR_non_healthy,folder)
        tsne(os.path.join(self.path,str(folder)),folder)
        plt.close("all")

        with open(self.path+"/"+str(folder)+"/log_"+str(folder)+".txt","w") as f:
            print(" With learning rate of: "+str(lr) +" On Test dataset we achieved an Overall Mean Average Precission of: "+ str(MAP) , file = f)
            print(" With learning rate of: "+str(lr) + ' On Test dataset we achieved an Overall Mean Reciprocal Rank of: ' + str(MRR), file=f)
            print("MAP & MRR for Healthy class is: "+str(MAP_per_class[0])+" & "+str(MRR_per_class[0]),file=f)
            print("MAP & MRR for Mild class is: "+str(MAP_per_class[1])+" & "+str(MRR_per_class[1]),file=f)
            print("MAP & MRR for Moderate class is: "+str(MAP_per_class[2])+" & "+str(MRR_per_class[2]),file=f)
            print("MAP & MRR for Severe class is: "+str(MAP_per_class[3])+" & "+str(MRR_per_class[3]),file=f)
            print("MAP & MRR for Proliferative class is: "+str(MAP_per_class[4])+" & "+str(MRR_per_class[4]),file=f)
            print("MAP & MRR for NON-Healthy class is: "+ str(MAP_non_healthy_val) +" & "+str(MRR_non_healthy_val), file=f)
            f.close()

    def quadratic_kappa(self,y, t, eps=1e-15):
      # Assuming y and t are one-hot encoded!
      num_scored_items = y.shape[0]
      num_ratings = y.shape[1]
      ratings_mat = np.tile(np.arange(0, num_ratings)[:, None],
                            reps=(1, num_ratings))
      ratings_squared = (ratings_mat - ratings_mat.T) ** 2
      weights = ratings_squared / (float(num_ratings) - 1) ** 2

      # We norm for consistency with other variations.
      y_norm = y / (eps + y.sum(axis=1)[:, None])

      # The histograms of the raters.
      hist_rater_a = y_norm.sum(axis=0)
      hist_rater_b = t.sum(axis=0)

      # The confusion matrix.
      conf_mat = np.dot(y_norm.T, t)

      # The nominator.
      nom = np.sum(weights * conf_mat)
      expected_probs = np.dot(hist_rater_a[:, None],
                              hist_rater_b[None, :])
      # The denominator.
      denom = np.sum(weights * expected_probs / num_scored_items)

      return 1 - nom / denom

    def one_hot(self,Y):
        N = len(Y)
        Y = Y.astype(np.int32)
        ind = np.zeros((N, 5))
        for i in range(N):
            ind[i, Y[i]] = 1
        return ind


    def fit(self,train_gen,val_gen, reg = 0.0005): # reg = 0.0005
        N = len(os.listdir("../Data/preprocessed_trainset/Images/"))

        im_width = 400

        for mi1, mo1, mo2 in self.conv2layer:
            c = Conv2Layer(mi1, mo1 , mo2,self.Initializer)
            self.conv_obj.append(c)
            im_width = int(np.ceil(im_width/2))
            mi1 = mo2

        for mi1, mo1, mo2, mo3 in self.conv3layer:
            c = Conv3Layer(mi1, mo1, mo2 , mo3, self.Initializer)
            self.conv_obj.append(c)
            im_width = int(np.ceil(im_width/2))
            mi1 = mo3

        tfX = tf.placeholder(tf.float32, shape=(None,400,400,3), name ="Input")
        tfY = tf.placeholder(tf.int64, shape=(None,), name="labels")
        lr = tf.placeholder(tf.float32, shape=[], name = "learning_rate")

        conv_out = self.forward(tfX)

        # For FC layers
        self.params = []
        self.FC = []
        conv_out_shape = conv_out.get_shape().as_list()
        conv_out_reshaped = tf.reshape(conv_out, [-1,np.prod(conv_out_shape[1:])])
        M1 = np.prod(conv_out_shape[1:])
        for M2 in self.FCLayer:
            h = FCLayer(M1, M2, self.Initializer)
            M1 = M2
            Z = h.forward(conv_out_reshaped)
            conv_out_reshaped = Z
            self.params+=h.params


        for c in self.conv_obj:
            self.params +=c.params



        embedding = Z
        tf.add_to_collection("embedding", embedding)


        W , b = self.Initializer.initialize_weights_bias(M2, 5)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        self.params+=[self.W,self.b]




        logits = tf.matmul(embedding,self.W)+ self.b # should be used when fc layers are used


        rcost = reg * sum([tf.nn.l2_loss(p) for p in self.params])
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tfY, name="cost"))
        loss = cost + rcost
        trainin_op = tf.train.AdamOptimizer(lr).minimize(loss)

        prediction = tf.argmax(logits,1)

        tf.add_to_collection("prediction", prediction)
        saver = tf.train.Saver(max_to_keep = None)

        init = tf.global_variables_initializer()

        n_batches  = N // self.batch_size
        epoch = 15
        N_val = len(os.listdir("../Data/preprocessed_devset/Images/"))
        n_batches_val = N_val // self.batch_size
        with tf.Session() as self.session:
            cost_list = []
            val_cost_list = []
            val_accu_list = []
            kappa_list = []
            self.session.run(init)
            for i in range(epoch):
                accumulated_cost = 0
                if i % 2 == 0:
                    fig = plt.figure()
                    plt.subplot(211)
                    plt.plot(val_cost_list,"-x", label ="Validation loss")
                    plt.plot(cost_list,"-x", label ="Training loss")
                    plt.legend()
                    plt.ylabel("cost/loss")
                    plt.xlabel("Epochs")
                    plt.subplot(212)
                    plt.plot(val_accu_list,"-x")
                    plt.ylabel("Accuracy")
                    plt.xlabel("Epochs")
                    fig1 = plt.figure()
                    plt.plot(kappa_list,"-x")
                    plt.xlabel("Epochs")
                    plt.ylabel("Kappa_score")
                    if not os.path.exists(os.path.join(self.path+"/"+str(i),"Plots")):
                        os.makedirs(os.path.join(self.path+"/"+str(i),"Plots"))
                    fig.savefig(os.path.join(self.path+"/"+str(i),"Plots")+"/vgg_train_"+str(i), transparent=False,bbox_inches = "tight" ,pad_inches=0)
                    fig1.savefig(os.path.join(self.path+"/"+str(i),"Plots")+"/kappa"+str(i), transparent=False,bbox_inches = "tight" ,pad_inches=0)
                    saver.save(self.session,self.path+"/"+str(i)+"/"+str(i))
                for j in range(n_batches): # default n_batches
                    X,Y,name_list=next(train_gen)
                    Y = Y.astype(np.int64)
                    self.session.run(trainin_op,feed_dict={tfX:X, tfY:Y, lr:self.lr})
                    if j % 500 ==0:
                        c =self.session.run(loss, feed_dict={tfX:X, tfY:Y})
                        p = self.session.run(prediction, feed_dict={tfX:X, tfY:Y})
                        accuracy=np.mean(p==Y)
                        accumulated_cost+=c
                        print("Training cost at epoch  %d ,batch %d of %d is %.3f , accuracy is %.3f" %(i, j, n_batches,c,accuracy))
                    if j % n_batches == 0:
                        val_cost = 0
                        val_prediction = np.zeros(n_batches_val*self.batch_size,dtype=np.int64)
                        val_label = np.zeros(n_batches_val*self.batch_size, dtype=np.int64)
                        for k in range(0,n_batches_val):
                            Xval , Yval , val_name = next(val_gen)
                            val_label[k*self.batch_size:(k+1)*self.batch_size,]=Yval
                            val_cost +=self.session.run(loss, feed_dict={tfX:Xval, tfY:Yval})
                            val_prediction[k*self.batch_size:(k+1)*self.batch_size,]=self.session.run(prediction,feed_dict={tfX:Xval})
                        val_pred_one_hot = self.one_hot(val_prediction)
                        val_label_one_hot = self.one_hot(val_label)
                        kappa = self.quadratic_kappa(val_pred_one_hot,val_label_one_hot)
                        val_accu = np.mean(val_prediction==val_label)
                        val_cost/=n_batches_val
                        val_cost_list.append(val_cost)
                        val_accu_list.append(val_accu)
                        kappa_list.append(kappa)
                        print("Validation cost at epoch %d ,is  %.3f, accuracy %.3f , kappa %.3f " %(i, val_cost, val_accu, kappa ))
                avg_train_loss = accumulated_cost/(n_batches//500)
                cost_list.append(avg_train_loss)
                print("Average Training cost at epoch %d is %.3f , accuracy is %.3f" %(i,avg_train_loss,accuracy))
            for k in range(0,epoch,2):
                self.make_plots(self.session,k, self.lr)


    def forward(self,X):
        Z = X
        for c in self.conv_obj:
            Z = c.forward(Z)
        for i in range(1):
            #fw = Z.get_shape().as_list()[1]
            mi = Z.get_shape().as_list()[3]
            final_conv = Conv2Layer(mi , 512, 512, self.Initializer) # find a suitable output feature map
            Z = final_conv.forward(Z)
        return Z

        """
        Z_shape = Z.get_shape().as_list()
        Z = tf.reshape(Z,[-1, np.prod(Z_shape[1:])])
        for h in self.FC:
            Z = h.forward(Z)
        return Z
        """

        """
        Z_shape = Z.get_shape().as_list()
        Z = tf.reshape(Z,[-1, np.prod(Z_shape[1:])])
        for h in self.FC:
            Z = h.forward(Z)
        return Z
        """

        """
        padding="VALID"
        for i in range(2):
            mi = Z.get_shape().as_list()[3]
            final_conv = Conv2Layer(mi , 2048, 2048, self.Initializer,fw, fw) # find a suitable output feature map
            Z = final_conv.forward_without_pool(Z, pad_type=padding)
            padding="SAME"
        return Z
        """


def main():
    path = "./Models/exp"
    batch_size=16
    train_gen = Generators(batch_size=batch_size).traindatagen()
    val_gen = Generators(batch_size=batch_size).valdatagen()
    Model = VGG([(3,64,64),(64,128,128)],[(128,256,256,256),(256,512,512,512),(512,512,512,512)],[128],Normal(), path,batch_size=batch_size)
    Model.fit(train_gen, val_gen)


if __name__ == "__main__":
    main()
