import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from evaluate import evaluate


def balanced_classes(X, Y):
    classes = 5
    class_no = []
    for i in range(classes):
        class_no.append(len(Y[Y==i]))
    min_no = min(class_no)
    data = []
    labels = []
    names_list = []
    count0 = 0
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    for index, label in enumerate(Y):
        if label==0:
            if count0<min_no:
                count0+=1
                data.append(X[index,:])
                labels.append(Y[index])
                #names_list.append(self.names_list[index])
        if label==1:
            if count1<min_no:
                count1+=1
                data.append(X[index,:])
                labels.append(Y[index])
                #names_list.append(self.names_list[index])
        if label==2:
            if count2<min_no:
                count2+=1
                data.append(X[index,:])
                labels.append(Y[index])
                #names_list.append(self.names_list[index])
        if label==3:
            if count3<min_no:
                count3+=1
                data.append(X[index,:])
                labels.append(Y[index])
                #names_list.append(self.names_list[index])
        if label==4:
            if count4<min_no:
                count4+=1
                data.append(X[index,:])
                labels.append(Y[index])
                #names_list.append(self.names_list[index])
    data = np.array(data)
    labels = np.array(labels)
    #self.names_list = np.array(names_list)
    return data , labels


def tsne(path,folder, balanced = False):
    classes =["Healthy", "Mild", "Moderate", "Severe","Proliferative"]
    data = np.load(os.path.join(path,"embedding.npy"))
    labels = np.load(os.path.join(path,"class_index.npy"))
    data = data / np.linalg.norm(data , ord = None , axis = 1)[:, None] 

    labels = labels.astype(np.int32)

    if balanced:
        data , labels = balanced_classes(data, labels)

    tsne = TSNE(learning_rate=1500) # lr 1200 on unbalanced and
    transformed_data = tsne.fit_transform(data)
    fig1 = plt.figure()
    for i , label in enumerate(classes):
        Z = transformed_data[labels==i]
        lab = labels[labels==i]
        plt.scatter(Z[:,0], Z[:,1], s=10, alpha=0.5,label=label)
        plt.legend()
    plt.title("T-SNE at epoch "+str(folder))
    plt.xlabel("X1")
    plt.ylabel("X2")
    fig1.savefig(path+"/tsne"+str(folder),transparent=False,bbox_inches = "tight" ,pad_inches=0)


def pca(path,balanced=False):
    classes =["Healthy", "Mild", "Moderate", "Severe","Proliferative"]
    data = np.load(os.path.join(path,"embedding.npy"))
    labels = np.load(os.path.join(path,"class_index.npy"))
    labels = labels.astype(np.int32)
    if balanced:
        data , labels = balanced_classes(data, labels)
    pca = PCA()
    reduced_data = pca.fit_transform(data)


    for i , label in enumerate(classes):
        Z = reduced_data[labels==i]
        lab = labels[labels==i]
        plt.scatter(Z[:,0], Z[:,1], s=10, alpha=0.5,label=label)
        plt.legend()
    plt.show()

    #plt.plot(pca.explained_variance_ratio_)
    #plt.show()

    # We now chose the number of dimentions that gives us 95-99 % of the variance
    cumulative = []

    last = 0
    for i , v in enumerate(pca.explained_variance_ratio_):
        cumulative.append(last+v)
        last = cumulative[-1]
        if last >= 0.99:
            break
    #plt.plot(cumulative)
    #plt.show()



if __name__ =="__main__":
    path = "./Models/VGG-balanced/6"
    #tsne(path,15,balanced = False)
    pca(path, balanced=False)
