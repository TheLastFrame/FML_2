import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import cohen_kappa_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from datapreprocessing import preprocess, preprocess_no_removes

seed = 61452651

random_state = 42

val_split_seed = 0

group = 'test-more-metrics'

config={
        "random_seed": seed,
        "data_dir": 'Dataset/Resampled/train',
        "val_dir": 'Dataset/Resampled/val',
        "test_dir": 'Dataset/Resampled/test',
        # "mp": False,
        # "gpu": tf.test.is_gpu_available(),
        "gpu": False,
        # "cpu_per_client": 16,
        # "gpu_per_client": 0,
        "num_clients": 1,
        "num_rounds": 2, #3,
        # "model_arch": 'eV2B0', #eV2L, eV2S, custom. eV2B0
        "model_type": 'NN', #NN, SVC, GradientBoostingClassifier, LogisticRegression
        "auto_rescaling":True,
        "epochs": 2, #5
        "fine_tune_at": 100,
        "lr": 'noCustomLr', # noCustomLr, 1e-X (5)
        "batch_size": 16, # 3080 can't handle 32 in normal mode (efficientNetV2L)
        "image_size": 256,
        "last_dropout": 0.20,
        # "color_mode": 'rgb', # rgb, grayscale - not implemented yet
        "pooling": None, # None, 'avg' or 'max' - currently only works with EfficientNet Models - old models are noPooling
        "featurewise_center": False,
        "featurewise_std_normalization": False,
        "do_data_augmentation": True, #@param {type:"boolean"}
        "zca":False,
        "zoom_range":[0.7, 1.0] #zoom_range
    }

def cohen_kappa(y_true, y_pred):
    y_true_classes = tf.argmax(y_true, 1)
    y_pred_classes = tf.argmax(y_pred, 1)

    # tf.print(tf.size(y_true_classes))

    #inverse and splitet if order to potentially save unnecessary calculations
    if len(y_true_classes) != 1: #cohen_kappa_score needs at least two different labels
        uniques, ids = tf.unique(y_true_classes)
        if len(uniques) == 1:
            uniques_pred, ids_pred = tf.unique(y_pred_classes)
            if len(uniques_pred) == 1 and uniques[0] == uniques_pred[0]:
                return tf.constant(0.0, dtype=tf.float64)
            return tf.py_function(lambda y_true_classes, y_pred_classes : cohen_kappa_score(y_true_classes.numpy(), y_pred_classes.numpy()), (y_true_classes, y_pred_classes), tf.double)
        return tf.py_function(lambda y_true_classes, y_pred_classes : cohen_kappa_score(y_true_classes.numpy(), y_pred_classes.numpy()), (y_true_classes, y_pred_classes), tf.double)
    else:
        return tf.constant(0.0, dtype=tf.float64)

def get_data(ds_type, all_columns = False, labels_as_int=True):
    if ds_type == 'A':
        train = pd.read_csv("./BankA_Train.csv")
        val = pd.read_csv("./BankA_Val.csv")
    elif ds_type == 'B':
        train = pd.read_csv("./BankB_Train.csv")
        val = pd.read_csv("./BankB_Val.csv")
    elif ds_type == 'C':
        train = pd.read_csv("./BankC_Train.csv")
        val = pd.read_csv("./BankC_Val.csv")
    else:
        raise ValueError("Invalid dataset type")
    
    if all_columns:
        train = preprocess_no_removes(train)
        val = preprocess_no_removes(val)
    else:
        train = preprocess(train)
        val = preprocess(val)

    if labels_as_int:
        train['income'] = train['income'].map({'>50K': 1, '<=50K': 0})
        val['income'] = val['income'].map({'>50K': 1, '<=50K': 0})

    X_train = train.drop('income', axis=1)
    y_train = train['income']
    X_val = val.drop('income', axis=1)
    y_val = val['income']

    return X_train, y_train, X_val, y_val

def get_test_data(ds_type, all_columns = False, labels_as_int=True):
    if ds_type == 'A':
        test = pd.read_csv("./BankA_Test.csv")
    elif ds_type == 'B':
        test = pd.read_csv("./BankB_Test.csv")
    elif ds_type == 'C':
        test = pd.read_csv("./BankC_Test.csv")
    else:
        test = pd.read_csv("./All_Banks_Test.csv")

    if all_columns:
        test = preprocess_no_removes(test)
    else:
        test = preprocess(test)

    if labels_as_int:
        test['income'] = test['income'].map({'>50K': 1, '<=50K': 0})

    X_test = test.drop('income', axis=1)
    y_test = test['income']

    return X_test, y_test

def get_labels(train_generator):

    return '\n'.join(sorted(train_generator.class_indices.keys()))

def get_model(config, input_shape, weights=None):
    model = keras.Sequential([
        layers.BatchNormalization(input_shape=input_shape),
        # layers.Dense(1028, activation='relu'),
        # layers.BatchNormalization(),
        # layers.Dropout(0.3),
        # layers.Dense(512, activation='relu'),
        # layers.BatchNormalization(),
        # layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'), 
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid'),
    ])
    
    opt = tf.keras.optimizers.Adam()
    if config["lr"] != 'noCustomLr':
        print('used optimizer without lr')
        opt = tf.keras.optimizers.Adam(config["lr"])

    loss='binary_crossentropy'
    metrics = ['binary_accuracy',
            #    tfa.metrics.CohenKappa(num_classes=train_generator.num_classes, weightage='quadratic'),
            #    tfa.metrics.F1Score(num_classes=train_generator.num_classes, threshold=0.5),
            tf.keras.metrics.Recall(thresholds=0.5),
            tf.keras.metrics.Precision(thresholds=0.5),
            # cohen_kappa
            ] #added all metrics here but recall is the most important for our usecase

    model.compile(loss=loss,
                optimizer = opt,
                metrics = metrics)
    
    if weights is not None:
        model.set_weights(weights)

    return model

def get_ml_model(model_type='SVC', parameters=None):
    model = None
    if model_type == 'SVC':
        model = SVC(random_state=random_state)
    elif model_type == 'GradientBoostingClassifier':
        model = GradientBoostingClassifier(random_state=random_state)
    elif model_type == 'LogisticRegression':
        model = LogisticRegression(random_state=random_state)
    else:
        raise ValueError("Invalid model type")

    if parameters is not None:
        model.set_params(parameters)
    
    return model