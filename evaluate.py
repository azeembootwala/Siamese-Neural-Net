import os
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)
import cv2
import copy
from sklearn.utils import shuffle


class evaluate(object):
    def __init__(self,path, K):
        self.K = K
        self.embedding = np.load(os.path.join(path,"embedding.npy"))
        class_indices = np.load(os.path.join(path,"class_index.npy"))
        self.Ytest = class_indices.astype(np.int64)
        self.names_list = np.load(os.path.join(path,"names.npy"))
        self.classes = ["Healthy","Mild", "Moderate", "Severe", "Proliferative"]
        self.embedding = self.embedding / np.linalg.norm(self.embedding , ord = None , axis = 1)[:, None] # Use only for classification


    def balanced_classes(self):
        classes = self.classes
        class_no = []
        for i in range(len(classes)):
            class_no.append(len(self.Ytest[self.Ytest==i]))
        min_no = min(class_no)
        embedding = []
        Ytest = []
        names_list = []
        count0 = 0
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        for index, label in enumerate(self.Ytest):
            if label==0:
                if count0<min_no:
                    count0+=1
                    embedding.append(self.embedding[index,:])
                    Ytest.append(self.Ytest[index])
                    names_list.append(self.names_list[index])
            if label==1:
                if count1<min_no:
                    count1+=1
                    embedding.append(self.embedding[index,:])
                    Ytest.append(self.Ytest[index])
                    names_list.append(self.names_list[index])
            if label==2:
                if count2<min_no:
                    count2+=1
                    embedding.append(self.embedding[index,:])
                    Ytest.append(self.Ytest[index])
                    names_list.append(self.names_list[index])
            if label==3:
                if count3<min_no:
                    count3+=1
                    embedding.append(self.embedding[index,:])
                    Ytest.append(self.Ytest[index])
                    names_list.append(self.names_list[index])
            if label==4:
                if count4<min_no:
                    count4+=1
                    embedding.append(self.embedding[index,:])
                    Ytest.append(self.Ytest[index])
                    names_list.append(self.names_list[index])
        self.embedding = np.array(embedding)
        self.Ytest = np.array(Ytest)
        self.names_list = np.array(names_list)


    def balanced_classes_random(self):
        classes = self.classes
        class_no = []
        for i in range(len(classes)):
            class_no.append(len(self.Ytest[self.Ytest==i]))
        min_no = min(class_no)
        embedding = []
        Ytest = []
        names_list = []
        idx = np.array([x for x in range(self.Ytest.shape[0])])
        Healthy = shuffle(idx[[(lambda i:x==0)(x) for i , x in enumerate(self.Ytest)]])[:min_no]
        Mild = shuffle(idx[[(lambda i:x==1)(x) for i , x in enumerate(self.Ytest)]])[:min_no]
        Moderate=shuffle(idx[[(lambda i:x==2)(x) for i , x in enumerate(self.Ytest)]])[:min_no]
        Severe=shuffle(idx[[(lambda i:x==3)(x) for i , x in enumerate(self.Ytest)]])[:min_no]
        Proliferative=shuffle(idx[[(lambda i:x==4)(x) for i , x in enumerate(self.Ytest)]])[:min_no]
        class_index_list = [Healthy , Mild , Moderate , Severe , Proliferative]
        for i in range(len(classes)):
            for index in class_index_list[i]:
                embedding.append(self.embedding[index,:])
                Ytest.append(self.Ytest[index])
                names_list.append(self.names_list[index])
        self.embedding = np.array(embedding)
        self.Ytest = np.array(Ytest)
        self.names_list = np.array(names_list)

    def healthy_disease_balance(self):
        classes = self.classes
        class_no = []
        for i in range(len(classes)):
            class_no.append(len(self.Ytest[self.Ytest==i]))
        min_no = min(class_no)
        embedding = []
        Ytest = []
        names_list = []
        idx = np.array([x for x in range(self.Ytest.shape[0])])
        Healthy = shuffle(idx[[(lambda i:x==0)(x) for i , x in enumerate(self.Ytest)]])[:min_no*4]
        Mild = shuffle(idx[[(lambda i:x==1)(x) for i , x in enumerate(self.Ytest)]])[:min_no]
        Moderate=shuffle(idx[[(lambda i:x==2)(x) for i , x in enumerate(self.Ytest)]])[:min_no]
        Severe=shuffle(idx[[(lambda i:x==3)(x) for i , x in enumerate(self.Ytest)]])[:min_no]
        Proliferative=shuffle(idx[[(lambda i:x==4)(x) for i , x in enumerate(self.Ytest)]])[:min_no]
        class_index_list = [Healthy , Mild , Moderate , Severe , Proliferative]
        for i in range(len(classes)):
            for index in class_index_list[i]:
                embedding.append(self.embedding[index,:])
                Ytest.append(self.Ytest[index])
                names_list.append(self.names_list[index])
        self.embedding = np.array(embedding)
        self.Ytest = np.array(Ytest)
        self.names_list = np.array(names_list)

    def calculate_distance(self,anchor_image):
        return np.sqrt(np.sum(np.square(anchor_image-self.embedding),axis = 1))
        #return np.sum(np.square(anchor_image-self.embedding),axis = 1) dont un comment


    def MRR(self):
        # We will compute the mean reciprocal rank
        #The score is reciprocal of the rank of the first relevant item
        reciprocal_rank = 0
        for i in range(self.embedding.shape[0]):
            query = self.embedding[i]
            query_label = self.Ytest[i]
            top_k_retrieved_idx , _ = self.sort(i)
            reterievel_labels = self.Ytest[top_k_retrieved_idx]
            min_idx = np.where(reterievel_labels==query_label)[0]
            if min_idx.size ==0:
                reciprocal_rank +=0 # 1/(K+1)
            else:
                rank = min(min_idx)
                reciprocal_rank += 1/(rank+1)

        mean_reciporcal_rank = reciprocal_rank/self.embedding.shape[0]
        return mean_reciporcal_rank


    def MAP(self):
        #Computes a mean average precission on embeddings
        AP = 0
        pre = 0  # this is the actual precission value
        for i in range(self.embedding.shape[0]):
            precission = 0
            query = self.embedding[i]
            query_label = self.Ytest[i]
            top_k_retrieved_idx , _ = self.sort(i)
            reterievel_labels = self.Ytest[top_k_retrieved_idx]
            min_idx = np.where(reterievel_labels==query_label)[0]
            pre+=min_idx.size
            if min_idx.size ==0:
                precission +=0
            else:
                for j , rank in enumerate(min_idx):
                    j+=1
                    precission+=(j/(rank+1))
                AP +=precission/min_idx.shape[0]
        MAP = AP/self.embedding.shape[0]
        MP = pre/(self.embedding.shape[0] * self.K)
        return MAP , MP

    def sort(self,query_index):
        anchor_image = self.embedding[query_index]
        distances = self.calculate_distance(anchor_image)
        top_k_indices  = np.argsort(distances)[1:self.K+1]
        top_k_distances = np.sort(distances)[1:self.K+1]
        return top_k_indices, top_k_distances

    def manual_check(self,query_index):
        image_path = "/media/azeem/Seagate Expansion Drive3/src/Data/preprocessed_devset/Images"
        query_image = cv2.imread(image_path+"/"+self.names_list[query_index])
        cv2.imshow("query_image :- " + str(self.classes[self.Ytest[query_index]]),query_image)
        cv2.waitKey(0)
        top_k_indices, top_k_distances= self.sort(query_index)
        for i in range(self.K):
            retrieved = cv2.imread(image_path+"/"+self.names_list[top_k_indices[i]])
            #cv2.namedWindow("reterived "+ str(i) + "  class "+str(self.classes[self.Ytest[top_k_indices[i]]]),cv2.WINDOW_NORMAL)
            #cv2.resizeWindow("reterived "+ str(i) + "  class "+str(self.classes[self.Ytest[top_k_indices[i]]]), 200,200)
            cv2.namedWindow("class: "+str(self.classes[self.Ytest[top_k_indices[i]]])+" " + str(i),cv2.WINDOW_NORMAL)
            cv2.resizeWindow("class: "+str(self.classes[self.Ytest[top_k_indices[i]]])+" " + str(i), 250,250)
            cv2.imshow("class: "+str(self.classes[self.Ytest[top_k_indices[i]]])+" " + str(i), retrieved)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Query_class:", self.classes[self.Ytest[query_index]])
        print("Retrieved_classes ", [self.classes[x] for x in self.Ytest[top_k_indices]])
        #print(self.Ytest[top_k_indices[3]])
        print("Retrieved_names" , self.names_list[top_k_indices])
        print("Respective distances ",top_k_distances)

    def per_class_stats(self):
        MAP = []
        MRR = []
        MP = []
        for i in range(5):
            mean_avg_precision , mean_precision = self.MAP_per_class(i)
            MAP.append(mean_avg_precision)
            MRR.append(self.MRR_per_class(i))
            MP.append(mean_precision)
        return MAP , MRR , MP


    def MAP_per_class(self,class_no):
        pre = 0
        AP = 0
        query_no = 0
        for i in range(self.embedding.shape[0]):
            query_label = self.Ytest[i]
            if query_label==class_no:
                precission = 0
                query_no+=1
                query = self.embedding[i]
                top_k_retrieved_idx,_ = self.sort(i)
                reterievel_labels = self.Ytest[top_k_retrieved_idx]
                min_idx = np.where(reterievel_labels==query_label)[0]
                pre += min_idx.size
                if min_idx.size ==0:
                    precission +=0
                else:
                    for j , rank in enumerate(min_idx):
                        j+=1
                        precission+=(j/(rank+1))

                    AP +=precission/min_idx.shape[0]
        MAP = AP/query_no
        MP = pre/(query_no*self.K)
        return MAP, MP


    def MRR_per_class(self,class_no):
        reciprocal_rank = 0
        query_no = 0
        for i in range(self.embedding.shape[0]):
            query_label = self.Ytest[i]
            if query_label==class_no:
                query_no+=1
                query = self.embedding[i]
                top_k_retrieved_idx , _ = self.sort(i)
                reterievel_labels = self.Ytest[top_k_retrieved_idx]
                min_idx = np.where(reterievel_labels==query_label)[0]
                if min_idx.size ==0:
                    reciprocal_rank +=0 # 1/(K+1)
                else:
                    rank = min(min_idx)
                    reciprocal_rank += 1/(rank+1)
        mean_reciporcal_rank = reciprocal_rank/query_no
        return mean_reciporcal_rank

    def MAP_non_healthy(self):
        pre = 0
        AP = 0
        query_no = 0
        non_zero_index = np.nonzero(self.Ytest)
        binary_class_labels = copy.deepcopy(self.Ytest)
        binary_class_labels[non_zero_index] = 1
        for i in range(self.embedding.shape[0]):
            query_label = binary_class_labels[i]
            if query_label == 1:
                query_no+=1
                precission = 0
                query = self.embedding[i]
                top_k_retrieved_idx , _ = self.sort(i)
                reterievel_labels = binary_class_labels[top_k_retrieved_idx]
                min_idx = np.where(reterievel_labels==query_label)[0]
                pre+= min_idx.size
                if min_idx.size ==0:
                    precission +=0
                else:
                    for j , rank in enumerate(min_idx):
                        j+=1
                        precission+=(j/(rank+1))
                    AP+=precission/min_idx.shape[0]
        MAP = AP/query_no
        MP = pre/(query_no*self.K)
        return MAP , MP

    def MRR_non_healthy(self):
        reciprocal_rank = 0
        query_no = 0
        non_zero_index = np.nonzero(self.Ytest)
        binary_class_labels = copy.deepcopy(self.Ytest)
        binary_class_labels[non_zero_index] = 1
        for i in range(self.embedding.shape[0]):
            query_label = binary_class_labels[i]
            if query_label == 1:
                query_no+=1
                query = self.embedding[i]
                top_k_retrieved_idx , _ = self.sort(i)
                reterievel_labels = binary_class_labels[top_k_retrieved_idx]
                min_idx = np.where(reterievel_labels==query_label)[0]
                if min_idx.size ==0:
                    reciprocal_rank +=0 # 1/(K+1)
                else:

                    rank = min(min_idx)
                    reciprocal_rank += 1/(rank+1)

        mean_reciporcal_rank = reciprocal_rank/query_no
        return mean_reciporcal_rank

    def referable_non_referable(self, type):
        pre = 0
        query_no = 0
        labels = np.array([0 if x==1 else x for x in self.Ytest])
        non_zero_index = np.nonzero(labels)
        binary_class_labels = copy.deepcopy(labels)
        binary_class_labels[non_zero_index] = 1
        for i in range(self.embedding.shape[0]):
            query_label = binary_class_labels[i]
            if query_label == type:
                query_no+=1
                query = self.embedding[i]
                top_k_retrieved_idx , _ = self.sort(i)
                reterievel_labels = binary_class_labels[top_k_retrieved_idx]
                min_idx = np.where(reterievel_labels==query_label)[0]
                pre+= min_idx.size
        MP = pre/(query_no*self.K)
        return MP , pre , query_no*self.K


    def cohens_kappa(self, TP , TP_FN , TN , TN_FP):
        FN = TP_FN - TP
        FP = TN_FP - TN
        Total_samples = TP + FN + TN +FP
        observed_agreement = (TP+TN)/ Total_samples

        first_part  = ((TP+FN)/Total_samples) * ((TP+FP)/Total_samples)
        second_part = ((TN+FN)/Total_samples) * ((FP+TN)/Total_samples)
        random_agreement = first_part + second_part
        kappa = (observed_agreement - random_agreement)/(1.0-random_agreement)
        return kappa


if __name__=="__main__":
    #path = "../Resnet/Models/Resnet50-128_longer/50"
    #path = "./Models/improved_margin/12"
    #path = "./Models/VGG-reduced10"
    #path = "./Models/VGG16-plain"
    #path = "./Models/VGG-margin0.4/8" # best till now
    path = "/media/azeem/Seagate Expansion Drive3/src/Triplet-Models/Batch_all-0.5_0.0001_b/14" # Best in Triplet batch all till now
    #path = "/media/azeem/Seagate Expansion Drive3/src/Triplet-Models/Batch_all_0.5_e-5/10"
    #path = "./Models/VGG-margin0.6/12"
    #path = "./Models/VGG-balanced/6"
    #path = "./Models/VGG-cross_entropy/8"
    labels = np.load(os.path.join(path,"class_index.npy"))
    labels = labels.astype(np.int64)
    #### Additions
    K= 6
    E = evaluate(path, K)
    #E.balanced_classes()
    #E.balanced_classes_random()
    E.healthy_disease_balance()
    labels = E.Ytest
    ### Additions end
    idx = np.array([x for x in range(labels.shape[0])])
    Healthy = idx[[(lambda i:x==0)(x) for i , x in enumerate(labels)]]

    Mild = idx[[(lambda i:x==1)(x) for i , x in enumerate(labels)]]
    Moderate=idx[[(lambda i:x==2)(x) for i , x in enumerate(labels)]]
    Severe=idx[[(lambda i:x==3)(x) for i , x in enumerate(labels)]]
    Proliferative=idx[[(lambda i:x==4)(x) for i , x in enumerate(labels)]]
    class_list = [Healthy,Mild, Moderate,Severe,Proliferative]

    class_picker = np.random.choice(len(class_list),p=np.array([0.2,0.2,0.2,0.2,0.2]))
    no = np.random.choice(class_list[class_picker])

    #print(no)
    E.manual_check(no)
    """
    MAP, MP = E.MAP() # funct 1
    print("Overall MAP achieved is: ",MAP)
    print("Ovearall MRR achieved is: ",E.MRR())
    MAP_per_class , MRR_per_class , MP_per_class = E.per_class_stats()
    print("MAP & MRR for Healthy class is: "+str(MAP_per_class[0])+" & "+str(MRR_per_class[0]))
    print("MAP & MRR for Mild class is: "+str(MAP_per_class[1])+" & "+str(MRR_per_class[1]))
    print("MAP & MRR for Moderate class is: "+str(MAP_per_class[2])+" & "+str(MRR_per_class[2]))
    print("MAP & MRR for Severe class is: "+str(MAP_per_class[3])+" & "+str(MRR_per_class[3]))
    print("MAP & MRR for Proliferative class is: "+str(MAP_per_class[4])+" & "+str(MRR_per_class[4]))
    MAP_non , MP_non = E.MAP_non_healthy() # funct 2
    print("MAP & MRR for NON-Healthy class is: "+ str(MAP_non) +" & "+str(E.MRR_non_healthy()))
    print("Overall Mean precision is ", MP)
    print("MP for Healthy class is: "+ str(MP_per_class[0]))
    print("MP for Mild class is: "+str(MP_per_class[1]))
    print("MP for Moderate class is: " +str(MP_per_class[2]))
    print("MP for Severe class is: " + str(MP_per_class[3]))
    print("MP for Proliferative class is:" + str(MP_per_class[4]))
    print("Mean Precision for NON-Healthy class is including Mild: ", MP_non)
    MP_healthy,TP,TP_FN = E.referable_non_referable(0) # funct 3
    MP_non_healthy , TN , TN_FP = E.referable_non_referable(1) # funt 4
    kappa = E.cohens_kappa(TP,TP_FN , TN , TN_FP) # funct 5
    print("Mean Precision for Non-Referable Diabetic Retinopathy (Sensitivity): ", MP_healthy)
    print("Mean Precision for Referable Diabetic Retinopathy:(Specificity)", MP_non_healthy)
    print("The Cohens Kappa Value is: ", kappa)
    """
