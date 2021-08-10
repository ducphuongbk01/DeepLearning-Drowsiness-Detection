import os
import cv2
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims = True)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')







model = tf.keras.applications.MobileNet()

base_input = model.layers[0].input
base_output = model.layers[-4].output

Flat_layer = layers.Flatten()(base_output)
final_output = layers.Dense(1)(Flat_layer)
final_output = layers.Activation('sigmoid')(final_output)

model = Model(inputs = base_input, outputs = final_output)

model.load_weights('./Model/Update_2/01.h5')

##EVALUATE BY PERCENTAGE OF ALL DATA
# optim = tf.keras.optimizers.SGD(learning_rate = 0.0001, momentum = 0.9)
# model.compile(optimizer = optim, loss = "binary_crossentropy", metrics = ['acc'])

# test_folder = "./Data/Test"

# test_datagen = ImageDataGenerator(rescale = 1.0/255.0)

# test_generator = test_datagen.flow_from_directory(test_folder, batch_size = 1, class_mode = 'binary', target_size = (224, 224))

# scores = model.evaluate_generator(test_generator, verbose = 1)

# print("Finished evaluate! \n")
# print("Results are: \n")
# print("%s %s: %.2f%%" % ("Model Update_2 01:",model.metrics_names[1], scores[1]*100))

# print("================================\n")
# print(scores)

##EVALUATE BY PERCENTAGE IN EACH CLASS AND DRAW CONFUSION MATRIX
test_data = []
dataDir = "./Data/Test/"
Classes = ["Close_eyes", "Open_eyes"]
for category in Classes:
    path = os.path.join(dataDir, category)
    class_num = Classes.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            new_array = cv2.resize(backtorgb, (224, 224))
            test_data.append([new_array, class_num])
        except Exception as e:
            pass


X = []
Y = []
for feature, label in test_data:
    X.append(feature)
    Y.append(label)

X = np.array(X).reshape(-1, 224, 224, 3)
X = X/255.0

Y = np.array(Y)

# pickle_out = open("X.pickle", "wb")
# pickle.dump(X, pickle_out)
# pickle_out.close()

# pickle_out = open("Y.pickle", "wb")
# pickle.dump(Y, pickle_out)
# pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("Y.pickle", "rb")
Y = pickle.load(pickle_in)

y_pred = model.predict(X)
y_pred_normalize = []
for output_model in y_pred:
    if output_model >= 0.5:
        out = "Open"
    else:
        out = "Close" 
    
    y_pred_normalize.append(out)

y_test = []
for output_label in Y:
    if output_label == 0:
        out_ = "Close"
    else:
        out_ = "Open"
    y_test.append(out_)

cnf_matrix = confusion_matrix(y_test, y_pred_normalize)
class_names = ["Close","Open"]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='MobileNet Model')

plt.show()