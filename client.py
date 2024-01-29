# %% [markdown]
# # Recognize Kidney Stones, Cysts & Tumors using Transfer Learning

# %%
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import random
import time
import os
import numpy as np

import tensorflow as tf

import flwr as fl
from flwr.common import Parameters, parameters_to_ndarrays, ndarrays_to_parameters

from sklearn.utils.class_weight import compute_class_weight
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

from utils import seed, group, config, get_model, get_data, get_test_data, get_labels


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
# wandb.login(key="f4f7847c29ad05b3b17541288561420a08e72a12")

# %%
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="fml-1",

#     group=group,
#     job_type="client",
#     name='client_'+str(client_number),

#     # track hyperparameters and run metadata with wandb.config
#     config=config
# )
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
    def __init__(self, model):
        self.model = model

    def get_parameters(self, config):
        print('starting parameters')
        # with open('log_parameters.txt', 'w') as f:
        #     f.write(str(config))
        # print(global_config)
        if self.model == None:
            self.model = get_model(global_config)
            print('new model created')
        print('ending parameters')
        return self.model.get_weights()

    def fit(self, parameters, config):
        print('starting fit')
        # with open('log.txt', 'w') as f:
        #     f.write(str(config))
        # print(global_config)
        train_generator, val_generator = get_data(global_config)
        if self.model == None:
            self.model = get_model(global_config, train_generator, parameters)
            print('new model created')
        else:
            self.model.set_weights(parameters)
        history = self.model.fit(train_generator,
                epochs=global_config["epochs"],
                steps_per_epoch = 10,
                callbacks=[
                            WandbMetricsLogger(log_freq=5),
                            WandbModelCheckpoint(group+'/'+client_number+'/wandb_save_models')
                            ],
                validation_data=val_generator)
        # wandb.run.summary['labels'] = get_labels(train_generator)

        # Get the last training accuracy
        accuracy = history.history['categorical_accuracy'][-1]
        recall = history.history['recall'][-1]
        precision = history.history['precision'][-1]
        cohen_kappa = history.history['cohen_kappa'][-1]
        print('ending fit')
        # return tuple([self.model.get_weights(), len(train_generator.filenames), {'categorical_accuracy': accuracy}]) #last is metrics
        return self.model.get_weights(), len(train_generator.filenames), {'categorical_accuracy': accuracy, "recall": recall, "precision": precision, "cohen_kappa": cohen_kappa} #last is metrics

    def evaluate(self, parameters, config):
        print('starting evaluate')
        test_generator = get_test_data(global_config)
        if self.model == None:
            self.model = get_model(global_config, test_generator,parameters)
            print('new model created')
        else:
            self.model.set_weights(parameters)
        eval = self.model.evaluate(test_generator, return_dict = True)
        #TODO: add metrics to wandb
        # wandb.log({'client_eval/loss': eval['loss']})
        # wandb.log({'client_eval/categorical_accuracy': eval['categorical_accuracy']})
        # wandb.log({'client_eval/recall': eval['recall']})
        # wandb.log({'client_eval/precision': eval['precision']})
        # wandb.log({'client_eval/cohen_kappa': eval['cohen_kappa']})# , commit=True
        print('ending evaluate')
        return eval['loss'], len(test_generator.filenames), eval


train_generator, val_generator = get_data(global_config)
client = IdentifAIClient(get_model(global_config, train_generator))
train_generator = None
val_generator = None
time.sleep(15)
fl.client.start_numpy_client(server_address="server:8080", client=client)
