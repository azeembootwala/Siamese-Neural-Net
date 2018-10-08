import numpy as np
import tensorflow as tf
import os
from Generators import Generators
from keras.preprocessing.image import ImageDataGenerator
np.set_printoptions(threshold=np.nan)

def infer_generator(session , model_path):
    batch_size = 16
    test_list = os.listdir("/cluster/azeem/Data/preprocessed_devset/Images/")
    N = len(test_list)
    n_batches = N // batch_size
    valgen = Generators(batch_size).valdatagen()
    embedding = tf.get_collection("embedding")[0]
    #model_path = "./Models/VGG16-plain/VGG16-plain"
    #imported_meta = tf.train.import_meta_graph(model_path+".meta")
    #imported_meta.restore(session,os.path.splitext(model_path)[0])
    #embedding = tf.get_collection("embedding")[0]
    #prediction = tf.get_collection("prediction")[0] # uncomment during classification network
    embeddings = []
    class_idx = []
    names = []
    for i in range(0,n_batches):
        Xtest_batch , Ytest_batch , test_name = next(valgen)
        class_idx= np.append(class_idx,Ytest_batch)
        embeddings.append(session.run(embedding,feed_dict={"Input:0":Xtest_batch}))
        names=np.append(names,test_name)

    names = np.array(names)
    class_idx= np.array(class_idx)
    embeddings = np.array(embeddings)
    embeddings= embeddings.reshape(np.prod(embeddings.shape[:-1]), embeddings.shape[-1])
    np.save(os.path.dirname(model_path)+"/"+"embedding", embeddings)
    np.save(os.path.dirname(model_path)+"/"+"class_index", class_idx)
    np.save(os.path.dirname(model_path)+"/"+"names", names)



def infer_generator_restore(session , model_path):
    batch_size = 16
    test_list = os.listdir("../Data/preprocessed_devset/Images")
    N = len(test_list)
    n_batches = N // batch_size
    valgen = Generators(batch_size).valdatagen()
    #model_path = "./Models/VGG16-plain/VGG16-plain"

    imported_meta = tf.train.import_meta_graph(model_path+".meta")
    imported_meta.restore(session,os.path.splitext(model_path)[0])
    embedding = tf.get_collection("embedding")[0]
    #prediction = tf.get_collection("prediction")[0] # uncomment during classification network
    embeddings = []
    class_idx = []
    names = []

    for i in range(0,n_batches):
        Xtest_batch , Ytest_batch , test_name = next(valgen)
        class_idx= np.append(class_idx,Ytest_batch)
        embeddings.append(session.run(embedding,feed_dict={"Input:0":Xtest_batch}))
        names=np.append(names,test_name)

    names = np.array(names)
    class_idx= np.array(class_idx)
    embeddings = np.array(embeddings)
    embeddings= embeddings.reshape(np.prod(embeddings.shape[:-1]), embeddings.shape[-1])
    np.save(os.path.dirname(model_path)+"/"+"embedding", embeddings)
    np.save(os.path.dirname(model_path)+"/"+"class_index", class_idx)
    np.save(os.path.dirname(model_path)+"/"+"names", names)



def infer_keras(session, path):
    #####             Keras stuff                                                           ##################
    datagen = ImageDataGenerator(rescale=1./255)
    val_generator = datagen.flow_from_directory("../Data/For_Keras-no-preprocess/Validation", target_size=(400,400),
                                                  batch_size = 16 ,classes=["Healthy","Mild-DR","Moderate-DR","Severe-DR","Proliferative-DR"],
                                                  class_mode = "sparse" , shuffle=False)
    ############                                        Keras stuff end                                              ###############


    batch_size = 16
    test_list = os.listdir("../Data/preprocessed_devset/Images")
    N = len(test_list)
    n_batches = N // batch_size
    valgen = Generators(batch_size).valdatagen()
    #model_path = "../Resnet/Models/Resnet50-plain/Resnet50-plain"
    model_path = path

    #with tf.Session() as session:
    imported_meta = tf.train.import_meta_graph(model_path+".meta")
    imported_meta.restore(session,os.path.splitext(model_path)[0])
    embedding = tf.get_collection("embedding")[0]
    embeddings = []
    class_idx = []
    names = []

    #for i in range(0,n_batches):
    for Xtest_batch , Ytest_batch in val_generator:
        if val_generator.batch_index == 0:
            continue
        #Xtest_batch , Ytest_batch , test_name = next(valgen)
        idx = (val_generator.batch_index - 1) * val_generator.batch_size
        file_names = [os.path.basename(i) for i in val_generator.filenames[idx : idx + val_generator.batch_size]]
        class_idx= np.append(class_idx,Ytest_batch)
        embeddings.append(session.run(embedding,feed_dict={"Input:0":Xtest_batch}))
        names=np.append(names,file_names)
        if val_generator.batch_index == val_generator.n // val_generator.batch_size:
            break

    names = np.array(names)
    class_idx= np.array(class_idx)
    embeddings = np.array(embeddings)
    embeddings= embeddings.reshape(np.prod(embeddings.shape[:-1]), embeddings.shape[-1])
    np.save(os.path.dirname(model_path)+"/"+"embedding", embeddings)
    np.save(os.path.dirname(model_path)+"/"+"class_index", class_idx)
    np.save(os.path.dirname(model_path)+"/"+"names", names)


if __name__ =="__main__":
    #pass
    session = tf.Session()
    infer_generator_restore(session , "./Models/VGG-balanced/8/8")
