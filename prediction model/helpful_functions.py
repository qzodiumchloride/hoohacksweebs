import cv2
import numpy as np
import os
from keras.utils import to_categorical
from skimage.transform import resize

weeb_dir = 'prediction model/moeimouto-faces/'

if not os.path.isdir(weeb_dir):
    print("Dataset not found.")
    quit()


def num_train_test(src_folder, img_folder):
    _, _, images = next(os.walk(src_folder+img_folder))
    num = len(images)-1
    num = int(num*0.8)
    return num


def test_train_split():
    X_train, Y_train, label_train = [], [], []
    X_test, Y_test, label_test = [], [], []

    X_train = np.zeros(shape=(1, 32, 32, 3))
    X_test = np.zeros(shape=(1, 32, 32, 3))

    idx = -1
    _, files, _ = next(os.walk(weeb_dir))
    for img_file in files:
        idx = idx + 1
        idy = 0
        _, _, images = next(os.walk(weeb_dir+img_file))
        for img in images:
            idy = idy + 1
            if(idy == 1):
                pass
            else:
                if idy < num_train_test(weeb_dir, img_file):
                    try:
                        image = cv2.imread(
                            os.path.join(weeb_dir+img_file, img))
                        image_resized = resize(image, (32, 32, 3))
                        image_resized = image_resized.reshape(1, 32, 32, 3)
                        # try:
                        #     if(image.shape == (160, 160, 3)):
                        #         image = image / 255
                        #         image = image.reshape(1, 160, 160, 3)
                        #     else:
                        #         x = 160
                        #         y = 160
                        #         if(image.shape[0] != 160):
                        #             x = 160 - image.shape[0]
                        #         if(image.shape[1] != 160):
                        #             y = 160 - image.shape[1]
                        #         temp = np.zeros(shape=(x, y, 3))
                        #         image = np.append(image, temp, axis=0)
                        #         image = image / 255
                        #         image = image.reshape(1, 160, 160, 3)

                        X_train = np.append(X_train, image_resized, axis=0)
                        Y_train.append(idx)
                        label_train.append(img_file)
                        # except:
                        #     pass
                    except:
                        pass
                else:
                    try:
                        image = cv2.imread(
                            os.path.join(weeb_dir+img_file, img))
                        image_resized = resize(image, (32, 32, 3))
                        image_resized = image_resized.reshape(1, 32, 32, 3)
                        # try:
                        #     if(image.shape == (160, 160, 3)):
                        #         image = image / 255
                        #         image = image.reshape(1, 160, 160, 3)
                        #     else:
                        #         x = 160
                        #         y = 160
                        #         if(image.shape[0] != 160):
                        #             x = 160 - image.shape[0]
                        #         if(image.shape[1] != 160):
                        #             y = 160 - image.shape[1]
                        #         temp = np.zeros(shape=(x, y, 3))
                        #         image = np.append(image, temp, axis=0)
                        #         image = image / 255
                        #         image = image.reshape(1, 160, 160, 3)

                        X_test = np.append(X_test, image_resized, axis=0)
                        Y_test.append(idx)
                        label_test.append(img_file)
                    except:
                        pass
                    # except:
                    #     pass

    # X_train = np.array(X_train)
    # X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    label_train = np.array(label_train)
    label_test = np.array(label_test)

    # X_trainx = []
    # i = 0
    # for pathdir in X_train:
    #     image = cv2.imread(os.path.join(weeb_dir+label_train[i], pathdir))
    #     X_trainx.append(image)
    #     i = i+1

    # X_testx = []
    # i = 0
    # for pathdir in X_test:
    #     image = cv2.imread(os.path.join(weeb_dir+label_test[i], pathdir))
    #     X_testx.append(image)
    #     i = i+1

    # X_trainx = np.array(X_trainx)
    # X_testx = np.array(X_testx)

    Y_train_one_hot = to_categorical(Y_train)
    Y_test_one_hot = to_categorical(Y_test)

    X_train = X_train[1:]/255
    X_test = X_test[1:]/255

    print(X_train.shape)
    print(Y_train_one_hot.shape)
    print(X_test.shape)
    print(Y_test_one_hot.shape)

    return (X_train, Y_train_one_hot, label_train), (X_test, Y_test_one_hot, label_test)
