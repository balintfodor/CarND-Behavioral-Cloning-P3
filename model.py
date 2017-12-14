import argparse
import glob
from tqdm import tqdm

import numpy as np
import pandas as pd
from skimage.io import imread
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import SGD, Adam

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
    parser.add_argument('--test', action='store_true')
    return parser.parse_args()

def find_csv_files(search_path):
    '''gets csv files recursively'''
    return glob.glob("{}/**/*.csv".format(search_path))

def image_path(csv_file_dir, abs_image_path_in_csv):
    '''generates an image path relative to its descriptor csv'''
    return str(csv_file_dir + '/IMG/' + abs_image_path_in_csv.split('/')[-1])

def import_dataframe(df, args):
    '''loads, formats and scales the data given in a dataframe'''
    data_x_list = []
    data_y_list = []
    for row in tqdm(df.iterrows(), total=len(df)):
        inputs = row[1][0:3].values
        targets = row[1][3:7].values
        targets = np.array(targets, ndmin=2)
        targets = np.repeat(targets, 3, axis=0)
        targets[1, 3] = np.maximum(targets[1, 3] - args.steering_compensation, -1.0)
        targets[2, 3] = np.minimum(targets[2, 3] + args.steering_compensation, 1.0)
        targets[:, 3] /= SPEED_DIVIDER
        for i, file_name in enumerate(inputs):
            data_x_list.append(imread(file_name, as_grey=True))
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
            df = df.head(256)
        data_x_table, data_y_table = import_dataframe(df, args)
        data_x_table_list.append(data_x_table)
        data_y_table_list.append(data_y_table)

    data_x = np.vstack(data_x_table_list)
    data_y = np.vstack(data_y_table_list)
    return data_x, data_y

def build_simple_model(args):
    '''builds a basic model for test functionality'''
    model = Sequential()
    model.add(Flatten(input_shape=(160, 320, )))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(4))
    sgd = SGD(lr=args.learning_rate)
    model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
    return model

def main():
    args = parse_args()
    X, y = import_data(args.search_path, args)
    X_train, X_val, y_train, y_val = \
        train_test_split(X, y,
                         train_size=0.7,
                         test_size=0.3,
                         shuffle=True,
                         random_state=args.seed)

    model = build_simple_model(args)
    model.fit(X_train, y_train,
              batch_size=args.batch_size,
              epochs=args.epochs,
              validation_data=(X_val, y_val))

    model.save(args.out)
    

if __name__ == "__main__":
    main()
