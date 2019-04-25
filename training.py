from PIL import Image
import cv2
import glob
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Flatten,Dense,Lambda, Conv2D, Cropping2D, Dropout
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.optimizers import Adam, SGD
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras import optimizers
import tensorflow.keras.backend as K
import os
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

path = os.getcwd()+"/ucf_sports_actions/ucf action"

# Function to crop the image according to ground truth
def crop_image(imgpath, gt1, gt2, gt3, gt4):
    imm = Image.open(imgpath, "r")
    return imm.crop((int(gt1), int(gt2), int(gt1)+int(gt3), int(gt2)+int(gt4)))

# Reading ground truth file to get dimensions and labels
def read_gt(path):
    file = open(path, "r")
    return file.read().split()


def files_cwd(dir):
    r = []
    for dirs in os.listdir(dir):
        r.append(dirs)
    return r


def makeFolders(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def resizeImages(image, resX, resY):
    return cv2.resize(image, (resX, resY))


def getY_train(im):
    b = os.path.split(os.path.dirname(im))[-2]
    c = os.path.dirname(b)
    d = os.path.basename(c)
    return labels[d]

def get_categorical_accuracy_keras(y_true, y_pred):
    return K.mean(K.equal(K.argmax(y_true, axis=1), K.argmax(y_pred, axis=1)))


def kerasModel(train_x, test_x, train_y, test_y):
    model = Sequential()

    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu", input_shape=(128, 64, 3)))

    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))

    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))

    model.add(Conv2D(64, (3, 3), strides=(2, 2), activation="relu"))

    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(64, activation="relu"))

    model.add(Dense(32, activation="relu"))

    model.add(Dense(1, activation="relu"))
    model.add(Activation('softmax'))
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])

    model.fit(train_x, train_y, epochs=5)

    # model.compile(loss='mse', optimizer=Adam(lr=0.01, decay=1e-6),metrics=[get_categorical_accuracy_keras])

    # model.fit(train_x,train_y, shuffle=True, nb_epoch=5)

    score = model.evaluate(test_x, test_y, verbose=0)

    print("Accuracy: %.2f%%" % (score[1] * 100))
    return score[1] * 100


def keras_LOO():
    path = os.getcwd() + "/ucf_sports_actions/ucf action"
    images = glob.glob(path + "/**" + "/**/jpeg/*.jpg")
    accDict = {}
    num_action_infolder = {}
    path = os.getcwd() + "/ucf_sports_actions/ucf action"
    val_label = labels
    val_label = {y: x for x, y in val_label.items()}

    for i in labels.keys():
        num_action_infolder[labels[i]] = len(files_cwd(path + "/" + i))

    for value, label in enumerate(labels):
        actionPath = path + "/" + label
        listdir = files_cwd(actionPath)

        for ld in listdir:
            test_x_img = glob.glob(path + "/" + label + "/" + ld + "/jpeg/*.jpg")
            if (len(test_x_img) == 0):
                continue
            train_x_img = []
            print("Label Running >>" + label)
            print("Folder Running >>" + ld)
            for i in images:
                if i not in test_x_img:
                    train_x_img.append(i)
            train_x = []
            train_y = []
            test_y = []
            test_x = []
            for x in train_x_img:
                train_y.append(getY_train(x))
            for y in test_x_img:
                test_y.append(getY_train(y))
            for i in train_x_img:
                train_x.append(cv2.imread(i))
            for j in test_x_img:
                test_x.append(cv2.imread(j))
            acc = kerasModel(np.array(train_x), np.array(test_x), np.array(train_y), np.array(test_y))
            if test_y[0] not in accDict.keys():

                accDict[test_y[0]] = acc
            else:
                accDict[test_y[0]] += acc
    for i in accDict:
        print("Accuracy of label " + val_label[i] + " is ")
        print(accDict[i] / num_action_infolder[i])


actions = files_cwd(path)
labels = {'Walk-Front': 0, 'Kicking-Side': 1, 'Golf-Swing-Front': 2, 'Riding-Horse': 3, 'Lifting': 4,
        'Golf-Swing-Side': 5, 'SkateBoarding-Front': 6, 'Swing-Bench': 7, 'Golf-Swing-Back': 8, 'Kicking-Front': 9, 'Run-Side': 10,
        'Diving-Side': 11, 'Swing-SideAngle': 12}

for i in actions:
    inActionPath = path + "/" + i  # in action
    folders_actions = files_cwd(inActionPath)
    for j in folders_actions:
        images_path = inActionPath + "/" + j  # inside 001
        # gt_path=inActionPath+"/"+j+"/gt" #inside gt
        files = files_cwd(images_path)  # getting all images
        if "jpeg" not in files:
            makeFolders(images_path + "/jpeg")
            item = ".jpg"
            if any(item in s for s in files):
                for f in files:
                    if os.path.isdir(images_path + "/gt"):
                        if f.endswith(".jpg"):
                            image = images_path + "/" + f
                            gt_path = images_path + "/gt/" + f.split(".")[0] + ".tif.txt"
                            readGt = read_gt(gt_path)
                            croppedImage = crop_image(image, readGt[0], readGt[1], readGt[2], readGt[3])
                            croppedImage.save(image)
                            cvim = cv2.imread(image)
                            reim = resizeImages(cvim, 64, 128)
                            cv2.imwrite(images_path + "/jpeg/" + f, reim)
                    else:
                        print("Gt do not exist for this " + images_path)

            else:
                print("Images not available in " + images_path)
        else:
            jpegPath = images_path + "/jpeg"
            jpgs = files_cwd(jpegPath)
            for x in jpgs:
                if ".jpg" in x:
                    im = cv2.imread(jpegPath + "/" + x)

                    reim = resizeImages(im, 64, 128)
                    cv2.imwrite(jpegPath + "/" + x, reim)


keras_LOO()

