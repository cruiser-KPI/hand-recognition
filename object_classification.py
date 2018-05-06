import os

# modify output level (0 - all logs, 1 - minus INFO logs, 2 - minus WARNING logs, 3 - minus ERROR logs)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tflearn
import data_loading

MODEL_FILE_PATH = 'pretrained_models/model.tfl.data-00000-of-00001'

INPUT_DATA_SIZE_X = data_loading.IMAGE_SIZE_X
INPUT_DATA_SIZE_Y = data_loading.IMAGE_SIZE_Y
OUTPUT_DATA_SIZE = data_loading.CLASSES_NUM

LABEL_MAP = {0: 'left', 1: 'right', 2: 'bad'}


def build_resnet():
    ''' Specify layers for ResNet and return constructed network '''

    net = tflearn.input_data(shape=[None, INPUT_DATA_SIZE_X, INPUT_DATA_SIZE_Y, 1])
    net = tflearn.conv_2d(net, 64, 3, activation='relu', bias=False)
    # Residual blocks
    net = tflearn.residual_bottleneck(net, 3, 16, 64)
    net = tflearn.residual_bottleneck(net, 1, 32, 128, downsample=True)
    net = tflearn.residual_bottleneck(net, 2, 32, 128)
    net = tflearn.residual_bottleneck(net, 1, 64, 256, downsample=True)
    net = tflearn.residual_bottleneck(net, 2, 64, 256)
    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')
    net = tflearn.global_avg_pool(net)
    # Regression
    net = tflearn.fully_connected(net, OUTPUT_DATA_SIZE, activation='softmax')
    net = tflearn.regression(net, optimizer='momentum',
                             loss='categorical_crossentropy',
                             learning_rate=0.1)
    return net


def get_trained_model():
    ''' Return trained model (if no pre-trained model, then also train it) '''

    model = tflearn.DNN(build_resnet(),
                        tensorboard_verbose=0, tensorboard_dir='tensorboard')
    if os.path.exists(MODEL_FILE_PATH):
        print('-'*80)
        print('Pretrained model was found.')
        model.load(os.path.splitext(MODEL_FILE_PATH)[0])
    else:
        print('-' * 80)
        print('Pretrained model was not found. Starting training:')
        images_train, labels_train, images_test, labels_test = data_loading.load_data()

        model.fit(images_train, labels_train, n_epoch=10,
                  validation_set=(images_test, labels_test),
                  snapshot_step=100, show_metric=True, run_id='convnet_hand_recognition')
        model.save(os.path.splitext(MODEL_FILE_PATH)[0])
    return model


if __name__ == '__main__':
    model = get_trained_model()
