import argparse
import glob, os
from tqdm import tqdm

import numpy as np
import pandas as pd
from skimage.io import imread
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Lambda, Dropout, BatchNormalization, Cropping2D, Concatenate, Input, Conv2D, Activation
from keras.optimizers import SGD, Adam
from keras.applications.mobilenet import MobileNet
from keras.applications.xception import Xception
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model

SPEED_DIVIDER = 30.0

def parse_args():
    '''arg parsing'''
    parser = argparse.ArgumentParser(description='Behavioral Cloning')
    parser.add_argument('search_path', type=str)
    parser.add_argument('--out', type=str, nargs='?', default='model.h5')
    parser.add_argument('--learning_rate', type=float, nargs='?', default=0.01)
    parser.add_argument('--batch_size', type=int, nargs='?', default=256)
    parser.add_argument('--epochs', type=int, nargs='?', default=100)
    parser.add_argument('--steering_compensation', type=float, nargs='?', default=0.2)
    parser.add_argument('--seed', type=int, nargs='?', default=42)
    parser.add_argument('--dropout', type=float, nargs='?', default=0.2)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--plot_model', action='store_true')
    return parser.parse_args()

def find_csv_files(search_path):
    '''gets csv files recursively'''
    return glob.glob("{}/**/*.csv".format(search_path))

def image_path(csv_file_dir, abs_image_path_in_csv):
    '''generates an image path relative to its descriptor csv'''
    path = os.path.join(csv_file_dir, 'IMG', os.path.basename(abs_image_path_in_csv))
    return str(os.path.realpath(path))

def import_dataframe(df, csv_dir, args):
    '''loads, formats and scales the data given in a dataframe'''
    data_x_list = []
    data_y_list = []
    for row in tqdm(df.iterrows(), total=len(df)):
        inputs = row[1][0:3].values
        targets = row[1][3:7].values
        targets = np.array(targets, ndmin=2)
        targets = np.repeat(targets, 3, axis=0)
        # left image -> should turn right
        targets[1, 0] = np.minimum(targets[1, 0] + args.steering_compensation, 1.0)
        # right image -> should turn left
        targets[2, 0] = np.maximum(targets[2, 0] - args.steering_compensation, -1.0)
        targets[:, 3] /= SPEED_DIVIDER
        # keep only the steering for now
        targets = targets[:, [0]]
        for i, file_name in enumerate(inputs):
            data_x_list.append(imread(image_path(csv_dir, file_name), as_grey=False))
            data_y_list.append(targets[i])

    data_x = np.array(data_x_list)
    data_y = np.array(data_y_list)
    return data_x, data_y

def import_data(search_path, args):
    '''loads, formats, scales data given by several csv paths'''
    files = find_csv_files(search_path)
    data_x_table_list = []
    data_y_table_list = []
    for file in files:
        print('processing {}'.format(file))
        df = pd.read_csv(file, header=None)
        if args.test:
            df = df.head(32)
        csv_dir = os.path.dirname(os.path.realpath(file))
        data_x_table, data_y_table = import_dataframe(df, csv_dir, args)
        data_x_table_list.append(data_x_table)
        data_y_table_list.append(data_y_table)

    data_x = np.vstack(data_x_table_list)
    data_y = np.vstack(data_y_table_list)
    return data_x, data_y

def append_mirrored_data(data_x, data_y):
    '''flipping the images with its steering value to populate the dataset'''
    mirrored_data_x = np.copy(data_x)
    print('adding flipped images')
    for i in tqdm(range(mirrored_data_x.shape[0]), total=mirrored_data_x.shape[0]):
        mirrored_data_x[i] = np.flip(mirrored_data_x[i], axis=1)
    mirrored_data_y = np.copy(data_y)
    mirrored_data_y[:, 0] *= -1.0
    return np.vstack((data_x, mirrored_data_x)), np.vstack((data_y, mirrored_data_y))

def build_simple_model(args):
    '''builds a basic model for test functionality'''
    model = Sequential()
    model.add(Lambda(lambda x: 1.0/255.0 * x, input_shape=(160, 320, 3)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1))
    sgd = SGD(lr=args.learning_rate)
    model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
    return model

def scale_image(x):
    import tensorflow as tf
    IMAGE_SCALE_FACTOR = 1.0 / 255.0
    return x * IMAGE_SCALE_FACTOR

def build_model(args):
    '''builds the advanced model'''
    inputs = Input(shape=(160, 320, 3))
    inputs_scaled = Lambda(scale_image)(inputs)

    sub_input_1 = Cropping2D(cropping=((0, 0), (0, 160)))(inputs_scaled)
    sub_input_2 = Cropping2D(cropping=((0, 0), (160, 0)))(inputs_scaled)

    # highly discouraged, but I had an SSL verification error
    # when tried to download the keras.application model
    # import ssl
    # ssl._create_default_https_context = ssl._create_unverified_context

    conv_model = MobileNet(include_top=False, input_shape=(160, 160, 3))
    for layer in conv_model.layers:
        layer.trainable = False

    conv_out_1 = conv_model(sub_input_1)
    conv_out_2 = conv_model(sub_input_2)

    merged_conv = keras.layers.concatenate([conv_out_1, conv_out_2], axis=2)

    top = merged_conv
    top = Dropout(args.dropout)(top)

    top = Conv2D(filters=256, kernel_size=(1, 1))(top)
    top = BatchNormalization()(top)
    top = Activation('relu')(top)
    top = Dropout(args.dropout)(top)

    top = Flatten()(top)

    top = Dense(256)(top)
    top = BatchNormalization()(top)
    top = Activation('relu')(top)
    top = Dropout(args.dropout)(top)

    top = Dense(64)(top)
    top = BatchNormalization()(top)
    top = Activation('relu')(top)

    top = Dense(1)(top)
    #top = Activation('sigmoid')(top)
    predictions = top

    model = Model(inputs=inputs, outputs=predictions)

    adam = Adam(lr=args.learning_rate)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])

    if args.plot_model:
        plot_model(model, to_file='model.png')
    model.summary()

    return model

def main():
    args = parse_args()
    X, y = import_data(args.search_path, args)
    X, y = append_mirrored_data(X, y)
    X_train, X_val, y_train, y_val = \
        train_test_split(X, y,
                         train_size=0.7,
                         test_size=0.3,
                         shuffle=True,
                         random_state=args.seed)

    model = build_model(args)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    checkpoint = ModelCheckpoint(args.out,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')
    model.fit(X_train, y_train,
              batch_size=args.batch_size,
              epochs=args.epochs,
              validation_data=(X_val, y_val),
              callbacks=[early_stopping, checkpoint])

if __name__ == "__main__":
    main()
