import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' 

# https://github.com/tensorflow/tensorflow/issues/53519
os.environ['TF_DEVICE_MIN_SYS_MEMORY_IN_MB'] = '256' 

import tensorflow as tf
import keras
from dataset import load_data
from model import get_generator, get_discriminator, get_gan
from train import train_gan
from pathlib import Path
from keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

SEED = 225
keras.utils.set_random_seed(SEED)

gpus = tf.config.experimental.list_physical_devices('GPU')
print(f'Tensorflow Version: {tf.__version__}')
gpus = tf.config.experimental.get_device_details(gpus[0])
print(f'Device: {gpus["device_name"]}\nCompute Capability: {gpus["compute_capability"]}')


def main():
    lat_dim = 100
    num_classes = 10
    epochs = 250
    db_name = 'cifar10'
    channels = 3 if db_name == 'cifar10' else 1
    image_shape = (32, 32, channels)
    
    save_path = Path(f'state/{db_name}')
    save_path.mkdir(parents=True, exist_ok=True)
    
    g_model = get_generator(lat_dim=lat_dim, num_out_channels=channels, num_classes=num_classes)
    d_model = get_discriminator(image_shape=image_shape, num_classes=num_classes)
    
    g_model.summary()
    d_model.summary()
    
    keras.utils.plot_model(d_model, save_path / 'architecture_discriminator.png', show_shapes=True)
    keras.utils.plot_model(g_model, save_path / 'architecture_generator.png', show_shapes=True)
    gan_model = get_gan(g_model, d_model)
    
    
    ds, label_mapping = load_data(db_name)
    
    train_gan(
        g_model, 
        d_model, 
        gan_model, 
        lat_dim,
        ds,
        label_mapping,
        n_epochs=epochs,
        half_batch_size=512,
        save_path=save_path
    )
    

if __name__ == "__main__":
    main()