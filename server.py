import os
import random
import time
import flwr as fl
from flwr.common import Parameters, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
import numpy as np
import tensorflow as tf
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from utils import seed, group, config, get_model, get_data, get_test_data, get_labels

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

train_generator, val_generator = get_data(config)
model = get_model(config, train_generator)
train_generator = None
val_generator = None

def get_config(_unused):
    return config


#server evaluation
def get_eval_fn(model):    
    test_generator = get_test_data(config)
    # model = get_model(config, test_generator)

    def evaluate(server_round, parameters, configuration) :
        model.set_weights(parameters)
        eval = model.evaluate(test_generator,
            verbose=1,
            # batch_size = 1,
            # workers = 8,
            # use_multiprocessing = True,
            return_dict = True)

        # wandb.log({"eval/loss":eval['loss']}, step=server_round)
        # wandb.log({"eval/categorical_accuracy": eval['categorical_accuracy']}, step=server_round)
        # wandb.log({"eval/recall": eval['recall']}, step=server_round)
        # wandb.log({"eval/precision": eval['precision']}, step=server_round)
        # wandb.log({"eval/cohen_kappa": eval['cohen_kappa']}, step=server_round)

        return eval['loss'], eval

    return evaluate



def get_base_weights(model):
    return ndarrays_to_parameters(model.get_weights())



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