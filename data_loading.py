import os
import time
import numpy as np
import cv2

cv2.setUseOptimized(True)
cv2.setNumThreads(4)

DATA_DIR = 'data'
IMAGE_DIRS = ['LeftHand50x50', 'RightHand50x50', 'Bad50x50']
CLASSES_NUM = len(IMAGE_DIRS)

ADDITIONAL_IMAGES_NUM = 5

IMAGE_SIZE_X = 50
IMAGE_SIZE_Y = 50


def shuffled(a, b):
    ''' Shuffle two arrays simultaneously '''

    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def rotateImage(image, angle):
    ''' Return image rotated by specified angle '''

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def load_data():
    ''' Load and preprocess data. Return image and label data splitted into training and testing set '''

    # 0 - left hand, 1 - right hand, 2 - bad image
    start = time.time()

    image_data = []
    image_labels = []

    for index, image_dir in enumerate(IMAGE_DIRS):
        dir_path = os.path.join(DATA_DIR, image_dir)

        image_list = [path for path in os.listdir(dir_path) if '.jpg' in path]
        for image_path in image_list:
            image = cv2.imread(os.path.join(dir_path, image_path), 0)
            image_data.append(image / 255.0)

            # augment data with rotated images
            for i in range(ADDITIONAL_IMAGES_NUM):
                angle = np.random.randint(-180, 180)
                new_image = image.copy()
                rotated = rotateImage(new_image, angle)
                image_data.append(rotated / 255.0)
        total_number = len(image_list) * (ADDITIONAL_IMAGES_NUM + 1)

        label_indices = np.empty(total_number, dtype=np.int)
        label_indices.fill(index)
        one_hot_labels = np.zeros((total_number, CLASSES_NUM))
        one_hot_labels[np.arange(total_number), label_indices] = 1
        image_labels.append(one_hot_labels)

    image_data = np.concatenate([row for row in image_data])
    image_data = np.reshape(image_data, (-1, IMAGE_SIZE_X, IMAGE_SIZE_Y, 1))

    image_labels = np.concatenate([row for row in image_labels])
    image_labels = np.reshape(image_labels, (-1, CLASSES_NUM))

    image_data, image_labels = shuffled(image_data, image_labels)

    train_test_percentage = 0.8
    fraction = int(len(image_data) * train_test_percentage)

    print('-'*80)
    print('Data was successfully loaded. Time elapsed:', time.time() - start)
    print('Total number of images:', len(image_data))
    print('Train images number:', fraction, 'Test images number:', len(image_data) - fraction)
    print('-' * 80)
    return image_data[:fraction], image_labels[:fraction], image_data[fraction:], image_labels[fraction:]


if __name__ == '__main__':
    images_train, labels_train, images_test, labels_test = load_data()
