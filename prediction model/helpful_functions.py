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
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    _, files, _ = next(os.walk(weeb_dir))
    for img_file in files:
        idy = 0
        _, _, images = next(os.walk(weeb_dir+img_file))
        for img in images:
            idy = idy + 1
            if idy < num_train_test(weeb_dir, img_file):
                X_train.append(img)
                Y_train.append(img_file)
            else:
                X_test.append(img)
                Y_test.append(img_file)

    X_train = np.array(X_train[1:])
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test[1:])

    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)

    return (X_train, Y_train), (X_test, Y_test)
