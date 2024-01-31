import os
import random
import time
import flwr as fl
from flwr.common import Parameters, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, log_loss, precision_score, recall_score
import tensorflow as tf
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from utils import get_ml_model, seed, group, config, get_model, get_data, get_test_data, get_labels

random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
tf.experimental.numpy.random.seed(seed)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(seed)

# Enable GPU growth in the main thread (the one used by the
# server to quite likely run global evaluation using GPU)
enable_tf_gpu_growth()

# wandb.login(key="f4f7847c29ad05b3b17541288561420a08e72a12")

test_dir = config["test_dir"]


# wandb.init(
#     # set the wandb project where this run will be logged
#     project="fml-1",
#     group=group,
#     job_type="server",    
#     # reinit=True,

#     # track hyperparameters and run metadata with wandb.config
#     config=config
# )


if config["model_type"] == 'NN':
    X_test, y_test = get_test_data('All', labels_as_int=True)
    input_shape = [X_test.shape[1]]
    model = get_model(config, input_shape)
else:
    model = get_ml_model(config["model_type"])

def get_config(_unused):
    return config


#server evaluation
def get_eval_fn(model):
    # model = get_model(config, test_generator)

    def evaluate_nn(server_round, parameters, configuration) :
        X_test, y_test = get_test_data('All', labels_as_int=True)
        X_test = X_test.astype('float32')
        y_test = y_test.astype('float32')
        model.set_weights(parameters)
        eval = model.evaluate(X_test, y_test,
            return_dict = True)

        # wandb.log({"eval/loss":eval['loss']}, step=server_round)
        # wandb.log({"eval/categorical_accuracy": eval['categorical_accuracy']}, step=server_round)
        # wandb.log({"eval/recall": eval['recall']}, step=server_round)
        # wandb.log({"eval/precision": eval['precision']}, step=server_round)
        # wandb.log({"eval/cohen_kappa": eval['cohen_kappa']}, step=server_round)

        return eval['loss'], eval
    
    def evaluate_ml(server_round, parameters, configuration):
        X_test, y_test = get_test_data('All')
        model.set_params(parameters)
        predictions_proba = model.predict_proba(X_test)

        # Use the stored predictions
        predictions = np.argmax(predictions_proba, axis=1)

        # Calculate the loss
        loss = log_loss(y_test, predictions_proba)
        
        # Metriken
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, pos_label='>50K')
        recall = recall_score(y_test, predictions, pos_label='>50K')
        f1 = f1_score(y_test, predictions, pos_label='>50K')
        kappa = cohen_kappa_score(y_test, predictions)

        return loss, {'loss': loss, 'accuracy': accuracy, "recall": recall, "precision": precision, "f1": f1, "cohen_kappa": kappa}


    if config["model_type"] == 'NN':
        return evaluate_nn
    else:
        return evaluate_ml



# def get_base_weights(model):
#     return ndarrays_to_parameters(model.get_weights())



# strategy = fl.server.strategy.FedAvg(eval_fn=get_eval_fn(model))
# strategy = fl.server.strategy.FedAvg(on_fit_config_fn=get_config, min_fit_clients=config["num_clients"], min_evaluate_clients=config["num_clients"],min_available_clients=config["num_clients"],on_evaluate_config_fn=get_config, evaluate_fn=get_eval_fn(model), initial_parameters=get_base_weights(model)) # initial_parameters = model.get_weights()
# strategy = fl.server.strategy.FedAvg(min_fit_clients=config["num_clients"], min_evaluate_clients=config["num_clients"],min_available_clients=config["num_clients"])
strategy = fl.server.strategy.FedAvg(min_fit_clients=config["num_clients"], min_evaluate_clients=config["num_clients"],min_available_clients=config["num_clients"], evaluate_fn=get_eval_fn(model))


start = time.time()

print('Starting at: {}'.format(time.strftime("%H:%M:%S", time.gmtime(start))))


fl.server.start_server(config=fl.server.ServerConfig(num_rounds= config["num_rounds"]), strategy=strategy)

end = time.time()
training_time = end - start
print('Training time: {}'.format(time.strftime(
    "%H:%M:%S", time.gmtime(training_time))))
# wandb.run.summary["training_time_sek"] = training_time
# wandb.run.summary["training_time"] = time.strftime("%H:%M:%S", time.gmtime(training_time))

print('DOOOOOOOOOOOOONE!!!!!!!!!!!')

# wandb.finish()