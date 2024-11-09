import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt


def plot_samples(g_model: keras.Model, latent_vector_plot, num_classes, epoch, label_mapping, count_per_class=10, save_path='figs'):
    plt.figure(figsize=(count_per_class * 2, num_classes * 2))
    
    k = 1
    for j in range(count_per_class):
        for i in range(num_classes):
            plt.subplot(count_per_class, num_classes, k)
            
            lat = tf.expand_dims(latent_vector_plot[i, j], 0)
            pred = g_model.predict([lat, tf.constant([i])], verbose=0)[0]
            pred = (pred + 1) / 2 * 255
            plt.imshow(pred.astype(np.uint8))
            plt.title(label_mapping[i])
            plt.axis("off")
            k += 1
    plt.savefig(save_path / f'output-{epoch:03}.png')
    
    
    

def generate_fake_samples(g_model: keras.Model, latent_dim, half_batch_size=128, num_classes=10):
    latent_vector = tf.random.normal((half_batch_size, latent_dim,), 0, 1)
    fake_labels = tf.random.uniform((half_batch_size,), 0, num_classes+1)
    
    fake_samples = g_model.predict([latent_vector, fake_labels], verbose=0)
    return fake_samples, fake_labels