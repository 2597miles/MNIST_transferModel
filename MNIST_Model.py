import datetime
import keras
import csv
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras import backend as K

#get current time
now = datetime.datetime.now
#params
batch_size = 128
num_channels = 7
epochs = 5
img_rows, img_cols = 28, 28
filters = 32
pool_size = 2
kernel_size = 3

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)

def getData():
    # load MNIST datasets
    (x_train, y_train), (x_test, y_test) = mnist.load_data('MNIST_data')

    # create two datasets one with digits below 7 and one with 7 and above
    x_train_0_6 = x_train[y_train < 7]
    y_train_0_6 = y_train[y_train < 7]
    x_test_0_6 = x_test[y_test < 7]
    y_test_0_6 = y_test[y_test < 7]

    x_train_7_9 = x_train[y_train >= 7]
    y_train_7_9 = y_train[y_train >= 7] -3  #change the labels below 7  eq:( 7,8,9) to (4,5,6)
    x_test_7_9 = x_test[y_test >= 7]
    y_test_7_9 = y_test[y_test >= 7] -3

    x_train_0_6 = x_train_0_6.reshape((x_train_0_6.shape[0],) + input_shape)    #reshape to (-1,28,28,1)
    x_test_0_6 = x_test_0_6.reshape((x_test_0_6.shape[0],) + input_shape)
    x_train_0_6 = x_train_0_6.astype('float32')/255.    #convert dytpe to float32
    x_test_0_6 = x_test_0_6.astype('float32')/255.

    x_train_7_9 = x_train_7_9.reshape((x_train_7_9.shape[0],) + input_shape)    #reshape to (-1,28,28,1)
    x_test_7_9 = x_test_7_9.reshape((x_test_7_9.shape[0],) + input_shape)
    x_train_7_9 = x_train_7_9.astype('float32')/255.    #convert dytpe to float32
    x_test_7_9 = x_test_7_9.astype('float32')/255.
    # one-hot
    y_train_0_6 = keras.utils.to_categorical(y_train_0_6, num_channels)
    y_test_0_6 = keras.utils.to_categorical(y_test_0_6, num_channels)
    y_train_7_9 = keras.utils.to_categorical(y_train_7_9, num_channels)
    y_test_7_9 = keras.utils.to_categorical(y_test_7_9, num_channels)

    print('--------------------------------------------')
    print('x_train_0_6 shape is:', x_train_0_6.shape)
    print('y_train_0_6 shape is:',y_train_0_6.shape)
    print('x_test_0_6 shape is:',x_test_0_6.shape)
    print('y_test_0_6 shape is:',y_test_0_6.shape)
    print('--------------------------------------------')
    print('x_train_7_9 shape is:', x_train_7_9.shape)
    print('y_train_7_9 shape is:',y_train_7_9.shape)
    print('x_test_7_9 shape is:',x_test_7_9.shape)
    print('y_test_7_9 shape is:',y_test_7_9.shape)
    print('--------------------------------------------')

    return(x_train_0_6,y_train_0_6,x_test_0_6,y_test_0_6,
            x_train_7_9,y_train_7_9,x_test_7_9,y_test_7_9)


def train_model(model, train, test):
    x_train,y_train,x_test,y_test = train[0],train[1],test[0],test[1]
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
 
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
 
    t = now()
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split=0.2)
    print('Training time: %s' % (now() - t))
    result = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', result[0])
    print('Test accuracy:', result[1])

def feature_layer ():
    '''
    To define feature_layer:
    Conv + relu + Conv + relu + pooling + dropout
    '''
    feature_layer = [
        Conv2D(filters, kernel_size,
            padding='same',
            input_shape=input_shape),
        Activation('relu'),
        Conv2D(filters, kernel_size),
        Activation('relu'),
        MaxPooling2D(pool_size=pool_size),
        Dropout(0.25),
        Flatten(),
    ]
    return feature_layer

def classification_layer():
    '''
    To define classification_layer:
    128-full-connected + relu + dropout + num_channels-full-connected + softmax
    '''
    classification_layer = [
        Dense(784),
        Activation('relu',name='feature_output'),  #output layer
        Dropout(0.5),
        Dense(num_channels),
        Activation('softmax')
    ]
    return classification_layer

def Write_csv(Data,csv_file_name):
    #save feature_images and feature_labels as a csv file
    with open(csv_file_name,"w",newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(Data)
        print('Write all datas !')

def Load_csv(csv_file_name):
    #load featureData
    csv_reader = csv.reader(open(csv_file_name))
    featureData=[]
    for row in csv_reader:
        featureData.append(row)
    print("Read all datas !")
    return(featureData)

def cosine_distance(matrix1, matrix2):
    '''
    caculate cosin_dis
    '''
    matrix1_matrix2 = np.dot(matrix1, matrix2.transpose())#内积
    matrix1_norm = np.sqrt(np.multiply(matrix1, matrix1).sum(axis=1))#求模
    matrix1_norm = matrix1_norm[:, np.newaxis] #增加列数
    matrix2_norm = np.sqrt(np.multiply(matrix2, matrix2).sum(axis=1))
    matrix2_norm = matrix2_norm[:, np.newaxis]
    cosine_distance = np.divide(matrix1_matrix2, np.dot(matrix1_norm, matrix2_norm.transpose()))
    return cosine_distance

def compute_accuracy():
    # caculate accuracy
    cnt=0 #num of correct predicted-img
    for i in range(cosine_distance.shape[0]):
        loc = np.argmax(cosine_distance[i])  #get the location of max similarity
        index1,index2 = np.argmax(y_test_7_9[i]),np.argmax(y_train_7_9[loc])
        if index1 == index2:
            cnt+=1
    acc= cnt/y_test_7_9.shape[0]
    return(acc)

if __name__ == '__main__':
    #--------------------------------------------------------------------
    #step1: start training
    feature_layer = feature_layer()
    classification_layer = classification_layer()
    model = Sequential(feature_layer+classification_layer)      #create model
    (x_train_0_6,y_train_0_6,x_test_0_6,y_test_0_6,x_train_7_9,y_train_7_9,x_test_7_9,y_test_7_9) = getData()      #get data
    train_model(model,          # train model for 7-digit classification [0..6]
                (x_train_0_6, y_train_0_6),
                (x_test_0_6, y_test_0_6))
    for layer in feature_layer:        #freeze feature layer
        layer.trainable = False
    train_model(model,          #transfer learning: train model for 3-digit classification [7..9]
                (x_train_7_9, y_train_7_9),
                (x_test_7_9, y_test_7_9))
    feature_model = Model(inputs=model.input,              #output feature from the layer named 'feature_output'
                            outputs=model.get_layer('feature_output').output)
    feature_model.summary()
    feature_model.save('featureModel.h5')     #save the feature_model
    print('Feature Model saved !')

    #--------------------------------------------------------------------
    #step2: save the feature data
    feature_images = feature_model.predict(x_train_7_9)    #predict with train data from [7~9]
    Write_csv(Data=feature_images,csv_file_name="feature_images.csv")
    Write_csv(Data=y_train_7_9,csv_file_name="feature_labels.csv")

    #--------------------------------------------------------------------
    #step3: test the model
    prediction = feature_model.predict(x_test_7_9)  #generate predicted value

    #--------------------------------------------------------------------
    #step4: load the feature data and caculate cosine_distance
    cosine_distance = cosine_distance(prediction,feature_images)
    acc=compute_accuracy()
    print('=======================================================')
    print("After it`s been modified , the final accuracy of model is :", acc)