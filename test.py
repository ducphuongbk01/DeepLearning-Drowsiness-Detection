import os
import numpy as np
import cv2
import tensorflow as tf
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model

##EXTRACT IMAGE DATASET

# image_path = glob("E:/BK/Nam III/HK2/AI/Assignment/mrlEyes/*/*.png")
# save_open_path = "E:/BK/Nam III/HK2/AI/Assignment/Data/Open_eyes"
# save_close_path = "E:/BK/Nam III/HK2/AI/Assignment/Data/Close_eyes"
# save_path = "E:/BK/Nam III/HK2/AI/Assignment/Data"
# idx_open = 0
# idx_close = 0
# idx = 0
# for image in tqdm(image_path):
#     img = cv2.imread(image)
#     classes = image.split('_')[4]
#     if classes == '0':
#         cv2.imwrite(f"{save_close_path}/{idx_close}.png", img)
#         idx_close+=1
#     elif classes == '1':
#         cv2.imwrite(f"{save_open_path}/{idx_open}.png", img)
#         idx_open+=1 
#     else:
#         cv2.imwrite(f"{save_path}/{idx}.png", img)  
#         idx+=1

# print("Finished extract!")

##SEPARATE DATA
# save_path = "./Data/Test/"
# dir_path = "E:/BK/Nam III/HK2/AI/Assignment/Data/"
# for classes in tqdm(["Close_eyes", "Open_eyes"]):
#     save_path_loop = save_path + classes
#     dir_path_loop = glob(dir_path + classes + "/*.png")
#     idx_ = 4235
#     for i in range(41945,len(dir_path_loop),1):
#         img_path = dir_path + classes + f"/{i}.png"
#         img = cv2.imread(img_path)
#         cv2.imwrite(f"{save_path_loop}/{idx_}.png", img)
#         idx_+=1

    
# print("Finished!")


##TEST MODEL
# model = tf.keras.applications.MobileNet()

# base_input = model.layers[0].input
# base_output = model.layers[-4].output

# Flat_layer = layers.Flatten()(base_output)
# final_output = layers.Dense(1)(Flat_layer)
# final_output = layers.Activation('sigmoid')(final_output)

# model = Model(inputs = base_input, outputs = final_output)

# model.load_weights('./Model/Update_2/01.h5')

# img = cv2.imread("./Test/Model/open_4_glasses.png", cv2.IMREAD_GRAYSCALE)
# backtorgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
# new_img = cv2.resize(backtorgb, (224, 224))

# X_input = np.array(new_img).reshape(1, 224, 224, 3)

# X_input = X_input/255.0

# prediction = model.predict(X_input)

# if prediction >= 0.5:
#         out = "Open"
# else:
#         out = "Close" 

# print(out)


##DRAW CURVE
# epoch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# epoch_update_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


# train_loss = [0.5247,0.2501,0.2098,0.2026,0.1655,0.1700,0.1525,0.1364,0.1313,0.1273,0.1262,0.1362,0.1252,0.1361,0.1253,0.1343,0.1268,0.1339,0.1275,0.0996]
# val_loss = [0.3450,0.2947,0.2790,0.2537,0.2558,0.2325,0.2415,0.2361,0.2380,0.2443,0.2092,0.2181,0.2175,0.2260,0.2173,0.2198,0.2025,0.2179,0.2016,0.2086]
# train_acc = [0.7416,0.9075,0.9303,0.9306,0.9453,0.9499,0.9547,0.9577,0.9602,0.9574,0.9558,0.9549,0.9605,0.9515,0.9603,0.9532,0.9552,0.9555,0.9569,0.9641]
# val_acc = [0.8512,0.8656,0.8737,0.8763,0.8863,0.8988,0.8875,0.8931,0.8913,0.8919,0.9025,0.8981,0.9087,0.9013,0.9069,0.9019,0.9100,0.8988,0.9144,0.9194]




# train_loss_update_1 = [0.1235,0.1137,0.1128,0.1143,0.1123,0.1025,0.1280,0.1063,0.1100,0.1142,0.1337,0.1107,0.1115]
# val_loss_update_1 = [0.2140,0.2077,0.2169,0.2029,0.2260,0.1815,0.1892,0.2007,0.1867,0.2061,0.1790,0.1874,0.2061]
# train_acc_update_1 = [0.9615,0.9607,0.9599,0.9570,0.9597,0.9681,0.9509,0.9662,0.9657,0.9628,0.9499,0.9655,0.9579]
# val_acc_update_1 = [0.9062,0.9119,0.9056,0.9175,0.8981,0.9225,0.9237,0.9112,0.9237,0.9200,0.9237,0.9144,0.9119]

# for i in [train_loss_update_1, val_loss_update_1, train_acc_update_1, val_acc_update_1]:
#     for j, k in enumerate(i):
#         i[j] = k*100


# train_loss_update_2 = [0.1234,0.1034,0.1036,0.1163,0.1134,0.1060,0.0994,0.1124,0.1023,0.1006,0.0979,0.1034,0.1108,0.1081,0.1203,0.0912,0.0924,0.1033,0.1011,0.1018]
# val_loss_update_2 = [0.1726,0.1935,0.2060,0.1981,0.1826,0.1975,0.1862,0.1951,0.2154,0.1935,0.1823,0.1927,0.1889,0.2005,0.1746,0.1887,0.1883,0.1881,0.1856,0.1828]
# train_acc_update_2 = [0.9614,0.9683,0.9675,0.9549,0.9621,0.9588,0.9630,0.9594,0.9654,0.9683,0.9623,0.9639,0.9626,0.9663,0.9573,0.9669,0.9705,0.9638,0.9615,0.9672]
# val_acc_update_2 = [0.9300,0.9169,0.9119,0.9150,0.9200,0.9219,0.9044,0.9212,0.9131,0.9181,0.9206,0.9175,0.9175,0.9131,0.9219,0.9156,0.9212,0.9244,0.9219,0.9294]



# train_loss_update_3 = [0.1161,0.1048,0.1300,0.1308,0.1212,0.1120,0.1180,0.1130,0.1126,0.1129,0.0967,0.1034,0.1231,0.1186,0.1065,0.1174,0.1266,0.1098,0.1143,0.1097]
# val_loss_update_3 = [0.1923,0.2087,0.1878,0.1890,0.2075,0.1771,0.2053,0.1788,0.2081,0.1795,0.2043,0.2056,0.1806,0.2028,0.1882,0.1890,0.1813,0.1905,0.2040,0.1918]
# train_acc_update_3 = [0.9602,0.9664,0.9557,0.9547,0.9537,0.9607,0.9598,0.9618,0.9632,0.9631,0.9675,0.9657,0.9605,0.9582,0.9661,0.9577,0.9589,0.9664,0.9638,0.9599]
# val_acc_update_3 = [0.9081,0.9000,0.9244,0.9175,0.9062,0.9206,0.9050,0.9212,0.9094,0.9194,0.9144,0.9050,0.9250,0.9119,0.9206,0.9137,0.9206,0.9175,0.9075,0.9212]



# fig, (ax1, ax2) = plt.subplots(2,1,sharex = True, dpi = 120, figsize = (5,5))

# ax1.plot(epoch_update_1, train_acc_update_1, 'go-', label='Training')
# ax1.plot(epoch_update_1, val_acc_update_1, 'ro-', label='Validation')

# ax2.plot(epoch_update_1, train_loss_update_1, 'go-', label='Training')
# ax2.plot(epoch_update_1, val_loss_update_1, 'ro-', label='Validation')


# ax1.set_title("Accuracy")
# ax2.set_title("Loss")

# ax1.set_xlabel("Epoch"); ax2.set_xlabel("Epoch")
# ax1.set_ylabel("%"); ax2.set_ylabel("%")
# ax1.set_xlim(0,14); ax2.set_xlim(0,14)
# ax1.set_ylim(88,100); ax2.set_ylim(0,25)

# plt.tight_layout()
# plt.show()

x = ['a', 'b', 'c']

print('d' in x)