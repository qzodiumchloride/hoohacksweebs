import cv2
import glob
import numpy as np
import os

weeb_dir = 'prediction model/moeimouto-faces/'

if not os.path.isdir(weeb_dir):
    print("Dataset not found.")
    quit()


def num_train_test(src_folder, img_folder):
    _, _, images = next(os.walk(src_folder+img_folder))
    num = len(images)
    num = int(num*0.8)
    return num


def test_train_split():
    X_train, Y_train, label_train = [], [], []
    X_test, Y_test, label_test = [], [], []

    idx = 0
    _, files, _ = next(os.walk(weeb_dir))
    for img_file in files:
        idx = idx + 1
        idy = 0
        _, _, images = next(os.walk(weeb_dir+img_file))
        for img in images:
            idy = idy + 1
            if idy < num_train_test(weeb_dir, img_file):
                X_train.append(img)
                Y_train.append(idx)
                label_train.append(img_file)
            else:
                X_test.append(img)
                Y_test.append(idx)
                label_test.append(img_file)

    X_train = np.array(X_train[1:])
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    label_train = np.array(label_train[1:])
    label_test = np.array(label_test)

    X_trainx = []
    i = 0
    for pathdir in X_train:
        image = cv2.imread(os.path.join(weeb_dir+label_train[i], pathdir))
        X_trainx.append(image)
        i = i+1

    X_testx = []
    i = 0
    for pathdir in X_test:
        image = cv2.imread(os.path.join(weeb_dir+label_test[i], pathdir))
        X_testx.append(image)
        i = i+1

    X_trainx = np.array(X_trainx)
    X_testx = np.array(X_testx)

    return (X_trainx, Y_train, label_train), (X_testx, Y_test, label_test)
