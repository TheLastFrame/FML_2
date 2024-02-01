import pandas as pd
import numpy as np
from numpy.random import RandomState
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import cohen_kappa_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.utils import resample

from datapreprocessing import preprocess, preprocess_no_removes

seed = 61452651

random_state = 42

val_split_seed = 0

group = 'fml-2-nn-2'

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
        "num_clients": 3,
        "num_rounds": 10, #3,
        # "model_arch": 'eV2B0', #eV2L, eV2S, custom. eV2B0
        "model_type": 'NN', #NN, SVC, GradientBoostingClassifier, LogisticRegression
        "auto_rescaling":True,
        "epochs": 15, #5
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

def get_data(ds_type, all_columns = False, labels_as_int=True, balance_train=False):
    if ds_type == 'A':
        train = pd.read_csv("./BankA_Train.csv", index_col=0)
        val = pd.read_csv("./BankA_Val.csv", index_col=0)
    elif ds_type == 'B':
        train = pd.read_csv("./BankB_Train.csv", index_col=0)
        val = pd.read_csv("./BankB_Val.csv", index_col=0)
    elif ds_type == 'C':
        train = pd.read_csv("./BankC_Train.csv", index_col=0)
        val = pd.read_csv("./BankC_Val.csv", index_col=0)
    else:
        train = pd.read_csv("./All_Banks_Train.csv", index_col=0)
        val = pd.read_csv("./All_Banks_Val.csv", index_col=0)

    if balance_train:
        # Calculate the number of samples in each class
        num_class0 = len(train[train.income=='<=50K'])
        num_class1 = len(train[train.income=='>50K'])

        # Determine the majority and minority classes
        if num_class0 > num_class1:
            df_majority = train[train.income=='<=50K']
            df_minority = train[train.income=='>50K']
        else:
            df_majority = train[train.income=='>50K']
            df_minority = train[train.income=='<=50K']

        print('Majority class size: ', num_class0)

        # Get the mean amount of data between the income labels
        mean_samples = int((len(df_majority) + len(df_minority)) // 2)

        # Upsample minority class
        df_minority_upsampled = resample(df_minority, 
                                         replace=True,     # sample with replacement
                                         n_samples=mean_samples,    # to match majority class
                                         random_state=random_state) # reproducible results

        # Downsample majority class
        df_majority_downsampled = resample(df_majority, 
                                           replace=False,    # sample without replacement
                                           n_samples=mean_samples,     # to match minority class
                                           random_state=random_state) # reproducible results

        # Combine majority class with upsampled minority class
        train = pd.concat([df_majority_downsampled, df_minority_upsampled])
    
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
        test = pd.read_csv("./BankA_Test.csv", index_col=0)
    elif ds_type == 'B':
        test = pd.read_csv("./BankB_Test.csv", index_col=0)
    elif ds_type == 'C':
        test = pd.read_csv("./BankC_Test.csv", index_col=0)
    else:
        test = pd.read_csv("./All_Banks_Test.csv", index_col=0)

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
        model = set_model_params(model, parameters, model_type)
    
    return model


def get_model_params(model, model_type='SVC'):
    if model_type == 'NN':
        return model.get_weights()
    elif model_type == 'SVC':
        params = []

        # Check if the model has been fitted
        if hasattr(model, 'dual_coef_'):
            params.extend([
                model.support_,
                model.support_vectors_,
                model.n_support_,
                model.dual_coef_,
                model.coef0,
            ])
            if model.decision_function_shape in ['ovo', 'ovr'] and len(model.classes_) == 2:
                params.append(model.intercept_)
            params.extend([
                model.fit_status_,
                model.probA_,
                model.probB_
            ])
        else:
            params.extend([
                model.C,
                model.degree,
                model.gamma,
                model.coef0,
                model.shrinking,
                model.probability,
                model.tol,
                model.cache_size,
                model.max_iter,
                model.break_ties,
                model.random_state,
            ])

        return params
    elif model_type == 'GradientBoostingClassifier': #TODO: check if this is correct (untested)
        params = [
            model.estimators_,
            model.feature_importances_,
            model.oob_improvement_,
            model.estimators_,
        ]
        return params
    elif model_type == 'LogisticRegression': #TODO: check if this is correct (untested)
        params = [
            model.coef_,
            model.intercept_,
        ]
        return params
    else:
        raise ValueError("Invalid model type")
    
def set_model_params(model, params, model_type='SVC'):
    if model_type == 'NN':
        model.set_weights(params)
    elif model_type == 'SVC':
        if hasattr(model, 'dual_coef_'):
            model.support_, model.support_vectors_, model.n_support_, model.dual_coef_, model.coef0 = params[:5]
            if model.decision_function_shape in ['ovo', 'ovr'] and len(model.classes_) == 2:
                model.intercept_ = params[5]
                params = params[6:]
            else:
                params = params[5:]
            model.fit_status_, model.probA_, model.probB_ = params
        else:
            # Ensure that the 'C' parameter is a float
            model.C = float(params[0])
            # Ensure that the 'degree' parameter is an integer
            model.degree = int(params[1])
            # Ensure that the 'gamma' parameter is a string among {'scale', 'auto'} or a float
            if str(params[2]) in ['scale', 'auto']:
                model.gamma = str(params[2])
            else:
                try:
                    model.gamma = float(params[2])
                except ValueError:
                    raise ValueError("Invalid value for 'gamma'. It must be either 'scale', 'auto' or a float.")
            # Ensure that the 'coef0' parameter is a float
            model.coef0 = float(params[3])
            # Ensure that the 'shrinking' parameter is a boolean
            model.shrinking = bool(params[4])
            # Ensure that the 'probability' parameter is a boolean or a numpy boolean
            model.probability = params[5] if isinstance(params[5], (bool, np.bool_)) else bool(params[5])
            # Ensure that the 'tol' parameter is a float
            model.tol = float(params[6])
            # Ensure that the 'cache_size' parameter is a float
            model.cache_size = float(params[7])
            model.max_iter = int(params[8])
            # Ensure that the 'break_ties' parameter is a boolean
            model.break_ties = bool(params[9])
            # Ensure that the 'random_state' parameter is an integer, a numpy random state, or None
            if isinstance(params[10], (int, np.integer)) or params[10] is None or isinstance(params[10], RandomState):
                model.random_state = params[10]
            else:
                try:
                    model.random_state = int(params[10])
                except ValueError:
                    raise ValueError("Invalid value for 'random_state'. It must be an integer, a numpy random state, or None.")
    elif model_type == 'GradientBoostingClassifier':
        model.estimators_, model.feature_importances_, model.oob_improvement_, model.estimators_ = params
    elif model_type == 'LogisticRegression':
        model.coef_, model.intercept_ = params
    else:
        raise ValueError("Invalid model type")

    return model