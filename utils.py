import tensorflow as tf
from sklearn.metrics import cohen_kappa_score

seed = 61452651

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
        "model_arch": 'eV2B0', #eV2L, eV2S, custom. eV2B0
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

    if len(y_true_classes) == 1: #cohen_kappa_score needs at least two different labels
        return tf.constant(0.0, dtype=tf.float64)
    return tf.py_function(lambda y_true_classes, y_pred_classes : cohen_kappa_score(y_true_classes.numpy(), y_pred_classes.numpy(), labels=[0,1,2,3,4,5,6,7]), (y_true_classes, y_pred_classes), tf.double)

def get_data(config):
    datagen_kwargs = dict()

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center = config["featurewise_center"],
        featurewise_std_normalization = config["featurewise_std_normalization"],
        zca_whitening=config["zca"],
        **datagen_kwargs)

    val_generator = datagen.flow_from_directory(
        config["val_dir"],
        target_size=(config["image_size"], config["image_size"]),
        batch_size=config["batch_size"])

    if config["do_data_augmentation"]:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            zoom_range=config["zoom_range"],
            **datagen_kwargs)
    else:
        train_datagen = datagen
    
    train_generator = train_datagen.flow_from_directory(
        config["data_dir"],
        shuffle=True,
        target_size=(config["image_size"], config["image_size"]),
        batch_size=config["batch_size"])
    
    return train_generator, val_generator

def get_test_data(config):
    
    imageDataGen = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center = config["featurewise_center"],
        featurewise_std_normalization = config["featurewise_std_normalization"],
        zca_whitening=config["zca"])

    test_generator = imageDataGen.flow_from_directory(
            config["test_dir"],
            target_size=(config["image_size"], config["image_size"]),
            shuffle = False,
            class_mode='categorical',
            batch_size=1)
    
    return test_generator

def get_labels(train_generator):

    return '\n'.join(sorted(train_generator.class_indices.keys()))


def get_model(config, train_generator, weights=None):
    IMG_SHAPE = (config["image_size"], config["image_size"], 3)

    base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(  # efficient net does not need any resizing, hast its own resizing layer at the beginning
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax',
        include_preprocessing=config["auto_rescaling"]
    )

    # Un-freeze the top layers of the model
    base_model.trainable = True

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:config["fine_tune_at"]]:
        layer.trainable =  False

    # Add a classification head
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Dropout(config["last_dropout"]),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(train_generator.num_classes,
                                activation='softmax') #, dtype='float32' // train_generator.num_classes if activation != 'sigmoid' else 1
    ])

    opt = tf.keras.optimizers.Adam()
    if config["lr"] != 'noCustomLr':
        print('used optimizer without lr')
        opt = tf.keras.optimizers.Adam(config["lr"])

    loss='categorical_crossentropy'
    metrics = ['categorical_accuracy',
            #    tfa.metrics.CohenKappa(num_classes=train_generator.num_classes, weightage='quadratic'),
            #    tfa.metrics.F1Score(num_classes=train_generator.num_classes, threshold=0.5),
            tf.keras.metrics.Recall(thresholds=0.5),
            tf.keras.metrics.Precision(thresholds=0.5),
            cohen_kappa] #added all metrics here but recall is the most important for our usecase


    model.compile(loss=loss,
                optimizer = opt,
                metrics = metrics)
    
    if weights is not None:
        model.set_weights(weights)

    return model