# %% [markdown]
# # Recognize Kidney Stones, Cysts & Tumors using Transfer Learning

# %%
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import random
import time
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, log_loss, precision_score, recall_score

import tensorflow as tf

import flwr as fl
from flwr.common import Parameters, parameters_to_ndarrays, ndarrays_to_parameters

from sklearn.utils.class_weight import compute_class_weight
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

from utils import seed, group, config, get_model, get_data, get_test_data, get_ml_model, set_model_params, get_model_params


# tf.set_random_seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
tf.experimental.numpy.random.seed(seed)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

client_number = os.getenv('CLIENT_NUMBER')
print('client_number: ', client_number)


# %%
wandb.login(key="f4f7847c29ad05b3b17541288561420a08e72a12")

# %%
wandb.init(
    # set the wandb project where this run will be logged
    project="fml-2",

    group=group,
    job_type="client",
    name='client_'+str(client_number),

    # track hyperparameters and run metadata with wandb.config
    config=config
)
global_config = config

# %% [markdown]
# ### Continue Train the model

# %%
# TODO: add early stopping
# TODO: reacitvate checkpoints
    
# earlystop = EarlyStopping(monitor='val_cohen_kappa',
#                           min_delta=.0001,
#                           patience=10,
#                           verbose=1,
#                           mode='auto',
#                           baseline=None,
#                           restore_best_weights=True)

#maybe this helps overcome our loss problems?
# reducelr = ReduceLROnPlateau(monitor='val_accuracy',
#                              factor=np.sqrt(.1),
#                              patience=5,
#                              verbose=1,
#                              mode='auto',
#                              min_delta=.0001,
#                              cooldown=0,
#                              min_lr=0.0000001)

# %%
class IdentifAIClient(fl.client.NumPyClient):
    def __init__(self, model=None):
        self.model = model

    def get_parameters(self, config):
        print('starting parameters')
        if self.model == None:
            if global_config["model_type"] == 'NN':
                X_train, _y_train, _X_val, _y_val = get_data(str(client_number), labels_as_int=True)        
                input_shape = [X_train.shape[1]]
                self.model = get_model(global_config, input_shape)
                print('new model created')
                print('ending parameters')
                return self.model.get_weights()
            else: 
                self.model = get_ml_model(model_type=global_config["model_type"])
                print('new model created')
                print('ending parameters')
                return get_model_params(self.model, global_config["model_type"])
        else:
            print('ending parameters')
            if global_config["model_type"] == 'NN':
                return self.model.get_weights()
            else:
                return get_model_params(self.model, global_config["model_type"])

    def fit(self, parameters, config):
        print('starting fit')
        if global_config["model_type"] == 'NN':
            return self._fit_nn(parameters)
        else:
            return self._fit_ml(parameters)
    
    def _fit_nn(self, parameters):
        X_train, y_train, X_val, y_val = get_data(str(client_number), labels_as_int=True)   
        X_train = X_train.astype('float32')
        X_val = X_val.astype('float32')
        y_train = y_train.astype('float32')
        y_val = y_val.astype('float32')
        if self.model == None:
            input_shape = [X_train.shape[1]]
            self.model = get_model(global_config, input_shape, parameters)
            print('new model created')
        else:
            self.model.set_weights(parameters)
        history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=global_config["epochs"],
                # steps_per_epoch = 10,
                callbacks=[
                            WandbMetricsLogger(log_freq=5),
                            WandbModelCheckpoint(group+'/'+client_number+'/wandb_save_models')
                            ]
                )
        # wandb.run.summary['labels'] = get_labels(train_generator)

        # Get the last training accuracy
        accuracy = history.history['binary_accuracy'][-1]
        recall = history.history['recall'][-1]
        precision = history.history['precision'][-1]
        # cohen_kappa = history.history['cohen_kappa'][-1]
        print('ending fit')
        return self.model.get_weights(), len(X_train), {'binary_accuracy': accuracy, "recall": recall, "precision": precision}

    def _fit_ml(self, parameters):
        X_train, y_train, X_val, y_val = get_data(str(client_number))
        if self.model == None:
            self.model = get_ml_model(global_config["model_type"], parameters)
            print('new model created')
        else:
            self.model = set_model_params(self.model, parameters, global_config["model_type"])

        # Get the last training accuracy
        self.model.fit(X_train, y_train)
        
        # Vorhersagen
        predictions = self.model.predict(X_val)
        
        # Metriken
        accuracy = accuracy_score(y_val, predictions)
        precision = precision_score(y_val, predictions, pos_label=1)
        recall = recall_score(y_val, predictions, pos_label=1)
        f1 = f1_score(y_val, predictions, pos_label=1)
        kappa = cohen_kappa_score(y_val, predictions)
        
        print('ending fit')
        # Ergebnisse speichern

        params = get_model_params(self.model, global_config["model_type"])
        return params, len(X_train), {'accuracy': accuracy, "recall": recall, "precision": precision, "f1": f1, "cohen_kappa": kappa}


    def evaluate(self, parameters, config):
        print('starting evaluate')
        if global_config["model_type"] == 'NN':
            return self._evaluate_nn(parameters)
        else:
            return self._evaluate_ml(parameters)
        
    def _evaluate_nn(self, parameters):
        X_test, y_test = get_test_data(str(client_number), labels_as_int=True)
        X_test = X_test.astype('float32')
        y_test = y_test.astype('float32')
        if self.model == None:
            input_shape = [X_test.shape[1]]
            self.model = get_model(global_config, input_shape, parameters)
            print('new model created')
        else:
            self.model.set_weights(parameters)
        eval = self.model.evaluate(X_test, y_test, return_dict = True)
        #TODO: add metrics to wandb
        wandb.log({'client_eval/loss': eval['loss']})
        wandb.log({'client_eval/binary_accuracy': eval['binary_accuracy']})
        wandb.log({'client_eval/recall': eval['recall']})
        wandb.log({'client_eval/precision': eval['precision']})
        # wandb.log({'client_eval/cohen_kappa': eval['cohen_kappa']})# , commit=True
        print('ending evaluate')
        return eval['loss'], len(X_test), eval
    
    def _evaluate_ml(self, parameters):
        X_test, y_test = get_test_data(str(client_number))

        if self.model == None:
            self.model = get_ml_model(global_config["model_type"], parameters)
            print('new model created')
        else:
            self.model = set_model_params(self.model, parameters, global_config["model_type"])
        # Store the predictions in a variable
        predictions_proba = self.model.predict_proba(X_test)

        # Use the stored predictions
        predictions = np.argmax(predictions_proba, axis=1)

        # Calculate the loss
        loss = log_loss(y_test, predictions_proba)
        
        # Metriken
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, pos_label=1)
        recall = recall_score(y_test, predictions, pos_label=1)
        f1 = f1_score(y_test, predictions, pos_label=1)
        kappa = cohen_kappa_score(y_test, predictions)

        return loss, len(X_test), {'loss': loss, 'accuracy': accuracy, "recall": recall, "precision": precision, "f1": f1, "cohen_kappa": kappa}




if global_config["model_type"] == 'NN':
    X_train, y_train, X_val, y_val = get_data(str(client_number), labels_as_int=True)
    input_shape = [X_train.shape[1]]
    client = IdentifAIClient(get_model(global_config, input_shape))
else:
    client = IdentifAIClient(get_ml_model(global_config["model_type"]))
# train_generator = None
# val_generator = None
time.sleep(30)
fl.client.start_numpy_client(server_address="fmlsserver:8080", client=client)
