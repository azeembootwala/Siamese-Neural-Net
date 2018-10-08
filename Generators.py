import numpy as np
import os
import cv2
import pandas
from sklearn.utils import shuffle
from glob import glob



class Generators(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        pass

    def traindatagen(self):
        #type = Full / Reduced
        data = "/cluster/azeem/Data/preprocessed_trainset/Images/"
        csv_labels = "/cluster/azeem/Data/preprocessed_trainset/labels.csv"
        i = 0
        j = 0
        data_list = os.listdir(data)
        data_list = shuffle(data_list)
        max_iter = int(len(data_list)/self.batch_size)*self.batch_size
        sample_shape = cv2.imread(data+data_list[1]).shape
        X_out = np.zeros((self.batch_size,*sample_shape),dtype = np.float32)
        Y_out = np.empty((self.batch_size,),dtype=np.int64)
        name_list = []
        labels = np.array(pandas.read_csv(csv_labels, sep=",", header=None))
        while True:
            if i==self.batch_size:
                i=0
                yield X_out/255 , Y_out , name_list
                name_list = []
            img = cv2.imread(data+data_list[j])
            label_idx= int(*(np.where(data_list[j]==labels[:,0])))
            X_out[i,]=img
            Y_out[i,]=labels[label_idx,1]
            name_list.append(data_list[j])
            if j == max_iter-1: # default max_iter and uncomment shuffle line
                j = -1
                data_list= shuffle(data_list)
            i+=1
            j+=1

    def valdatagen(self, balanced=False):
        dev_dir = "/cluster/azeem/Data/preprocessed_devset/Images/"
        image_list = os.listdir(dev_dir)
        if balanced:
            labels = np.array(pandas.read_csv(os.path.dirname(os.path.dirname(dev_dir))+"/labels.csv", sep=",", header=None))
            #idx = np.array([int(*(np.where(x==labels[:,0])))for x in image_list])
            idx = np.array([x for x in range(labels.shape[0])])
            Healthy = idx[[(lambda i:x==0)(x) for i , x in enumerate(labels[:,1])]]
            Mild = idx[[(lambda i:x==1)(x) for i , x in enumerate(labels[:,1])]]
            Moderate=idx[[(lambda i:x==2)(x) for i , x in enumerate(labels[:,1])]]
            Severe=idx[[(lambda i:x==3)(x) for i , x in enumerate(labels[:,1])]]
            Proliferative=idx[[(lambda i:x==4)(x) for i , x in enumerate(labels[:,1])]]
            class_names = [Healthy , Mild ,Moderate , Severe , Proliferative]
            min_no = min([len(x) for x in class_names])
            image_list = []
            for i in range(len(class_names)):
                count=0
                while count<min_no:
                #for count in range(min_no):
                    index = np.random.choice(class_names[i])
                    a = int(*(np.where(index==class_names[i])))
                    class_names[i]=np.delete(class_names[i],a)
                    name = labels[index,0]
                    image_list.append(name)
                    count+=1
        image_list = shuffle(image_list)
        i = 0
        j = 0
        sample_shape = cv2.imread(dev_dir+image_list[0]).shape
        max_iter = int(len(image_list)/self.batch_size)*self.batch_size
        X = np.zeros((self.batch_size,*sample_shape),dtype=np.float32)
        Y = np.empty(self.batch_size, dtype=np.int64)
        labels = np.array(pandas.read_csv(os.path.dirname(os.path.dirname(dev_dir))+"/labels.csv", sep=",", header=None))
        name_list = []
        while True:
            if i==self.batch_size:
                i=0
                yield X/255 , Y , name_list
                name_list = []
            img = cv2.imread(dev_dir+image_list[j])
            label_idx= int(*(np.where(image_list[j]==labels[:,0])))
            X[i,]=img
            Y[i,]=labels[label_idx,1]
            name_list.append(os.path.basename(image_list[j]))
            if j == max_iter-1:# default max_iter
                j = -1
                #image_list= shuffle(image_list)
            i+=1
            j+=1

    def testdatagen(self):
        i=0
        j=0
        dev_dir = "/cluster/azeem/Data/preprocessed_testset/Images/"
        image_list = os.listdir(dev_dir)
        #image_list = shuffle(image_list)
        sample_shape = cv2.imread(dev_dir+image_list[0]).shape
        max_iter = int(len(image_list)/self.batch_size)*self.batch_size
        X = np.zeros((self.batch_size,*sample_shape),dtype=np.float32)
        Y = np.empty(self.batch_size, dtype=np.int64)
        labels = np.array(pandas.read_csv(os.path.dirname(os.path.dirname(dev_dir))+"/labels.csv", sep=",", header=None))
        name_list = []
        while True:
            if i==self.batch_size:
                i=0
                yield X/255 , Y , name_list
                name_list = []
            img = cv2.imread(dev_dir+image_list[j])

            label_idx= int(*(np.where(image_list[j]==labels[:,0])))
            X[i,]=img
            Y[i,]=labels[label_idx,1]
            name_list.append(os.path.basename(image_list[j]))
            if j == max_iter-1:# default max_iter
                j = -1
                #image_list= shuffle(image_list)
            i+=1
            j+=1

def main():
    gen = Generators(16)
    train_gen = gen.valdatagen()
    count = 0
    for i in range(2):
        X, Y, Name = next(train_gen)




if __name__ =="__main__":
    main()
