# This code is fine and gives best results. Do not touch this file and rather use it for reference
# This is the main file for contrastive loss
import numpy as np
import tensorflow as tf
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from util_vgg import Normal , Xavier
from Generators import Generators
from histogram_loss import histogram
from inference import infer_generator
from evaluate import evaluate
from manifold_vizualization import tsne


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
    #def forward_without_pool(self, X, pad_type="VALID"):
            #conv_out = tf.nn.conv2d(X,self.w1, strides=[1,1,1,1],padding=pad_type, name="conv1-")#+str(iter))
            #conv_out = tf.nn.bias_add(conv_out, self.b1, name ="bias1-")#+str(iter)
            #conv_out = tf.nn.relu(conv_out)
            #return conv_out

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
        self.W = tf.Variable(W.astype(np.float32),name="W-fc-")
        self.b = tf.Variable(b.astype(np.float32),name="b-fc-")
        self.params =[self.W,self.b]

    def forward(self, X):
        out = tf.matmul(X, self.W)+self.b
        return out


class VGG(object):
    def __init__(self, conv2layer, conv3layer, Initializer, batch_size, path, lr ):
        self.lr = lr
        self.path = path
        self.conv2layer = conv2layer
        self.conv3layer = conv3layer
        self.Initializer = Initializer
        self.batch_size = batch_size
        #self.lr = 0.0001#  default 0.0001
        self.conv_obj = []
        self.FC_obj = []
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

    def make_plots(self,session,folder, lr , bins):
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
            print(" With learning rate of: "+str(lr)+" & bin size of "+str(bins)+" On Test dataset we achieved an Overall Mean Average Precision of: "+ str(MAP) , file = f)
            print(" With learning rate of: "+str(lr) + ' On Test dataset we achieved an Overall Mean Reciprocal Rank of: ' + str(MRR), file=f)
            print("MAP & MRR for Healthy class is: "+str(MAP_per_class[0])+" & "+str(MRR_per_class[0]),file=f)
            print("MAP & MRR for Mild class is: "+str(MAP_per_class[1])+" & "+str(MRR_per_class[1]),file=f)
            print("MAP & MRR for Moderate class is: "+str(MAP_per_class[2])+" & "+str(MRR_per_class[2]),file=f)
            print("MAP & MRR for Severe class is: "+str(MAP_per_class[3])+" & "+str(MRR_per_class[3]),file=f)
            print("MAP & MRR for Proliferative class is: "+str(MAP_per_class[4])+" & "+str(MRR_per_class[4]),file=f)
            print("MAP & MRR for NON-Healthy class is: "+ str(MAP_non_healthy_val) +" & "+str(MRR_non_healthy_val), file=f)
            f.close()


    def fit(self, traingen, valgen, num_steps):
        N = len(os.listdir("/cluster/azeem/Data/preprocessed_trainset/Images/"))
        self.im_width = 400

        for mi1, mo1, mo2 in self.conv2layer:
            c = Conv2Layer(mi1, mo1 , mo2,self.Initializer)
            self.im_width = int(np.ceil(self.im_width/2))
            self.conv_obj.append(c)
            mi1 = mo2

        for mi1, mo1, mo2, mo3 in self.conv3layer:
            c = Conv3Layer(mi1, mo1, mo2 , mo3, self.Initializer)
            self.im_width  = int(np.ceil(self.im_width/2))
            self.conv_obj.append(c)
            mi1 = mo3



        tfX = tf.placeholder(tf.float32, shape=(None,400,400,3), name ="Input")
        tfY = tf.placeholder(tf.float32, shape=(None,), name="labels")
        #lr = tf.placeholder(tf.float32,shape=[],name="learning_rate")


        final_layer = self.forward(tfX) # embedded features


        ##################                            FC Layer                         #########################
        M1 = self.im_width * self.im_width * final_layer.get_shape().as_list()[3]
        M2 = 128 # Dimention of embedding Default 128

        Z_shape = final_layer.get_shape().as_list()
        FC_layer_reshaped = tf.reshape(final_layer,[-1,np.prod(Z_shape[1:])])
        FC_obj = FCLayer(M1 , M2, self.Initializer)

        embedding = FC_obj.forward(FC_layer_reshaped)

        ### Summing up parameters for regularization  #####
        """
        self.params = []
        for h in self.FC_obj:
            self.params+=h.params
        for c in self.conv_obj:
            self.params +=c.params
        """


        #embedding = tf.reshape(final_layer,[tf.shape(final_layer)[0],tf.shape(final_layer)[-1]])
        embedding = tf.nn.l2_normalize(embedding, axis  = 1)

        tf.add_to_collection("embedding", embedding)

        # Saver object was defined here
        saver = tf.train.Saver(max_to_keep = None)

        ####  We now have the normalized embeddings , we will now pair them against each other to create B*(B-1)
        #### combinations where B is the batch size.
        cost = histogram(num_steps).hist_loss(embedding, tfY)
        trainin_op = tf.train.AdamOptimizer(self.lr).minimize(cost) # current best 0.00001

        epoch = 21 # default = 13
        n_batches = N // self.batch_size
        init = tf.global_variables_initializer()
        val_loss_list = []
        train_loss_list =[]
        self.session = tf.Session()
        self.session.run(init)
        for j in range(epoch):
            LL_val = 0
            LL_train = 0
            if j % 2 == 0:
                fig = plt.figure()
                plt.plot(train_loss_list,"-x",label="train_cost_epoch "+str(j))
                plt.plot(val_loss_list,"-x",label="val_cost_epoch "+str(j))
                plt.xlabel("Epochs")
                plt.ylabel("Cost")
                plt.legend()
                if not os.path.exists(os.path.join(self.path+"/"+str(j),"Plots")):
                    os.makedirs(os.path.join(self.path+"/"+str(j),"Plots"))
                fig.savefig(os.path.join(self.path+"/"+str(j),"Plots")+"/vgg_train_"+str(j), transparent=False,bbox_inches = "tight" ,pad_inches=0)
                saver.save(self.session,self.path+"/"+str(j)+"/"+str(j))
                self.make_plots(self.session,j, self.lr, num_steps)
            for i in range(n_batches):
                X, Y, name = next(traingen)
                self.session.run(trainin_op, feed_dict={tfX:X, tfY:Y})
                if i % 500 == 0:
                    loss_train = self.session.run(cost,feed_dict={tfX:X,tfY:Y})
                    Xval , Yval , val_name = next(valgen)
                    loss_val  = self.session.run(cost,feed_dict={tfX:Xval,tfY:Yval})
                    LL_val+=loss_val
                    LL_train += loss_train
                    print(" Training loss at epoch %d of %d iteration %d of %d ,  is %.6f" %(j ,epoch-1,i,n_batches,loss_train))
                    print(" Validation loss at epoch %d of %d iteration %d of %d , is %.6f" %(j ,epoch-1,i,n_batches,loss_val))

            LL_val/=(n_batches//500) # change 100 to a number at which you want to keep the interval
            LL_train/=(n_batches//500) # change 100 to a number at which you want to keep the interval
            val_loss_list.append(LL_val)
            train_loss_list.append(LL_train)
            print("  At epoch %d of %d , average train_loss is %.6f" %(j ,epoch-1,LL_train))
            print("  At epoch %d of %d , average val_loss is %.6f" %(j ,epoch-1,LL_val))
        self.session.close()



    def forward(self,X):
        Z = X
        for c in self.conv_obj:
            Z = c.forward(Z)

        for i in range(1):
            #fw = Z.get_shape().as_list()[1]
            self.im_width = int(np.ceil(self.im_width/2))
            mi = Z.get_shape().as_list()[3]
            final_conv = Conv2Layer(mi , 512, 512, self.Initializer)
            Z = final_conv.forward(Z)
        return Z


def main():
    batch_size=24
    num_steps = 128 # then 100 , 200 , 400
    lr = 1e-5
    path = "../Histogram-Models/"+str(num_steps)+"_"+str(lr)
    Model = VGG([(3,64,64),(64,128,128)],[(128,256,256,256),(256,512,512,512),(512,512,512,512)],Normal(), batch_size , path, lr )
    traingen = Generators(batch_size=batch_size).traindatagen()
    valgen = Generators(batch_size=batch_size).valdatagen()
    Model.fit(traingen, valgen,num_steps)


if __name__ == "__main__":
    main()
