import tensorflow as tf
import keras
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path
import shutil
from utils import generate_fake_samples, plot_samples

def train_gan(
    g_model: keras.Model, 
    d_model: keras.Model, 
    gan_model: keras.Model, 
    latent_dim, 
    dataset: tf.data.Dataset,
    label_mapping,
    num_classes=10,
    n_epochs=100, 
    half_batch_size=128,
    start=0,
    save_path:Path=Path('saves')
):
    
    print(dataset.cardinality())
    dataset = dataset.batch(half_batch_size)
    latent_vector_plot = tf.random.normal((10, num_classes, latent_dim), 0, 1)
    fig_savepath = save_path / 'plot_preds'
    try:
        shutil.rmtree(fig_savepath)
    except:
        pass
    fig_savepath.mkdir(parents=True, exist_ok=True)
    
    
    for epoch in range(start, start + n_epochs):
        
        d_losses = []
        d_accs = []
        g_losses = []
        
        for batch in tqdm(dataset):
            real_images = batch[0]
            real_labels = batch[1]
            n_samples = batch[0].shape[0]
            
            y_real = tf.ones((n_samples,))
            
            d_model.trainable = True
            d_loss1, d_acc1 = d_model.train_on_batch([real_images, real_labels], y_real)
            
            fake_samples, fake_labels = generate_fake_samples(g_model, latent_dim, n_samples, num_classes)
            y_fake = tf.zeros((n_samples,))
            
            d_loss2, d_acc2 = d_model.train_on_batch([fake_samples, fake_labels], y_fake)
            
            d_model.trainable = False
            z_vector = tf.random.normal((n_samples * 2, latent_dim,), 0, 1)
            z_labels = tf.random.uniform((n_samples * 2,), 0, num_classes+1)
            y_gan = tf.ones((n_samples * 2,))
            
            g_loss = gan_model.train_on_batch([z_vector, z_labels], y_gan)
            
            g_losses.append(g_loss)
            d_losses.append(d_loss1)
            d_losses.append(d_loss2)
            d_accs.append(d_acc1)
            d_accs.append(d_acc2)

        print(f'Epoch {epoch+1}/{n_epochs + start}:')
        print(f'D_loss:\t{np.mean(d_losses):0.4f}\tD_acc:\t{np.mean(d_accs):0.4f}')
        print(f'G_loss:\t{np.mean(g_losses):0.4f}')
        
        
        if epoch % 2 == 0:
            plot_samples(g_model, latent_vector_plot, num_classes, epoch, label_mapping=label_mapping, count_per_class=10, save_path=fig_savepath)
    
        g_model.save(save_path / 'weights_generator.keras')
        d_model.save(save_path / 'weights_discriminator.keras')
        gan_model.save(save_path / 'weights_gan.keras')