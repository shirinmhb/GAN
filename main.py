from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models import Model
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Input, Flatten, Dense
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.metrics import roc_auc_score
from keras import backend as K

def recall_m(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  recall = true_positives / (possible_positives + K.epsilon())
  return recall

def precision_m(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + K.epsilon())
  return precision

def f1_score_m(y_true, y_pred):
  precision = precision_m(y_true, y_pred)
  recall = recall_m(y_true, y_pred)
  return 2*((precision*recall)/(precision+recall+K.epsilon()))

size_img = 256
def check_AcGan():
  generated_covid = os.listdir('Generated/covid')
  generated_non_covid = os.listdir('Generated/non covid')
  xTestGenerated = np.zeros(shape=(len(generated_covid)+len(generated_non_covid), size_img, size_img, 3))
  yTestGenerated = np.zeros(shape=(len(generated_covid)+len(generated_non_covid), 2))
  for i in range(len(generated_covid)):
    path_img = generated_covid[i]
    if path_img[0] == '.':
      continue
    print('Generated/covid/'+path_img)
    img = image.load_img('Generated/covid/'+path_img, target_size=(size_img, size_img), color_mode='grayscale')
    x = image.img_to_array(img)
    xTestGenerated[i] = x
    yTestGenerated[i] = np.array([1,0])
  for i in range(len(generated_non_covid)):
    path_img = generated_non_covid[i]
    if path_img[0] == '.':
      continue
    print('Generated/non covid/'+path_img)
    img = image.load_img('Generated/non covid/'+path_img, target_size=(size_img, size_img), color_mode='grayscale')
    x = image.img_to_array(img)
    xTestGenerated[i+len(generated_covid)] = x
    yTestGenerated[i+len(generated_covid)] = np.array([0,1])
  return (xTestGenerated, yTestGenerated)


def read_samples():
  x = np.zeros(shape=(size_img, size_img, 3))
  train_covid = os.listdir('COVID,Non COVID-CT Images/train/COVID')
  train_not_covid = os.listdir('COVID,Non COVID-CT Images/train/Non-COVID')
  test_covid = os.listdir('COVID,Non COVID-CT Images/test/COVID')
  test_not_covid = os.listdir('COVID,Non COVID-CT Images/test/Non-COVID')
  num_train_covid, num_train_not_covid, num_test_covid, num_test_not_covid = len(train_covid), len(train_not_covid), len(test_covid), len(test_not_covid)
  train_shape = (num_train_covid+num_train_not_covid, x.shape[0], x.shape[1], x.shape[2])
  xTrain, yTrain = np.zeros(shape=(train_shape)), np.zeros(shape=(num_train_covid+num_train_not_covid,2))
  test_shape = (num_test_covid+num_test_not_covid, x.shape[0], x.shape[1], x.shape[2])
  xTest, yTest = np.zeros(shape=(test_shape)), np.zeros(shape=(num_test_covid+num_test_not_covid,2))
  for i in range(len(train_covid)):
    path_img = train_covid[i]
    img = image.load_img('COVID,Non COVID-CT Images/train/COVID/'+path_img, target_size=(size_img, size_img), color_mode='grayscale')
    x = image.img_to_array(img)
    xTrain[i] = x
    yTrain[i] = np.array([1,0])

  for i in range(len(train_not_covid)):
    path_img = train_not_covid[i]
    img = image.load_img('COVID,Non COVID-CT Images/train/Non-COVID/'+path_img, target_size=(size_img, size_img), color_mode='grayscale')
    x = image.img_to_array(img)
    xTrain[i+num_train_covid] = x
    yTrain[i+num_train_covid] = np.array([0,1])
  
  for i in range(len(test_covid)):
    path_img = test_covid[i]
    img = image.load_img('COVID,Non COVID-CT Images/test/COVID/'+path_img, target_size=(size_img, size_img), color_mode='grayscale')
    x = image.img_to_array(img)
    xTest[i] = x
    yTest[i] = np.array([1,0])

  for i in range(len(test_not_covid)):
    path_img = test_not_covid[i]
    img = image.load_img('COVID,Non COVID-CT Images/test/Non-COVID/'+path_img, target_size=(size_img, size_img), color_mode='grayscale')
    x = image.img_to_array(img)
    xTest[i+num_test_covid] = x
    yTest[i+num_test_covid] = np.array([0,1])

  print(xTrain.shape, xTest.shape, yTrain.shape, yTest.shape)
  return (xTrain, yTrain, xTest, yTest)



(xTestGan, yTestGan) = check_AcGan()
(xTrain, yTrain, xTest, yTest) = read_samples()
n = int(0.2 * len(xTrain))
xTest2 = xTest[:n]
yTest2 = yTest[:n]
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
model_vgg16_conv.summary()
input = Input(shape=(size_img, size_img, 3),name = 'image_input')
output_vgg16_conv = model_vgg16_conv(input)
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(2, activation='softmax', name='predictions')(x)
mlp = Model(inputs = input, outputs = x)
mlp.summary()
batch_size = 200
epochs = 10
mlp.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['acc',f1_score_m,precision_m, recall_m])
mlp.fit(xTrain, yTrain, batch_size=batch_size, epochs=epochs, validation_split=0.1)
loss, accuracy, f1_score, precision, recall = mlp.evaluate(xTest, yTest, verbose=0)
print("test results:")
print("loss:",loss,"accuracy:",accuracy,"f1 score:", f1_score,"precision:",precision,"recall:",recall)
yPred = mlp.predict(xTest)

XGan = np.concatenate((xTest2, xTestGan), axis=0)
YGan = np.concatenate((yTest2, yTestGan), axis=0)
loss, accuracy, f1_score, precision, recall = mlp.evaluate(XGan, YGan, verbose=0)
print("test gan results:")
print("loss:",loss,"accuracy:",accuracy,"f1 score:", f1_score,"precision:",precision,"recall:",recall)
yPredGan = mlp.predict(XGan)


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(yTest[:, i], yPred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(yTest.ravel(), yPred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 1
plt.plot(fpr[1], tpr[1], color='darkorange',
         lw=lw, label='AUC = %0.2f' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()




fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(YGan[:, i], yPredGan[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(YGan.ravel(), yPredGan.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 1
plt.plot(fpr[1], tpr[1], color='darkorange',
         lw=lw, label='AUC = %0.2f' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()