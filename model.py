import tensorflow as tf
import keras


def get_discriminator(image_shape, num_classes):
    input_label = keras.Input(shape=(1,))
    x = keras.layers.Embedding(num_classes, 100)(input_label)
    shape = image_shape[0] * image_shape[1]
    x = keras.layers.Dense(shape)(x)
    x = keras.layers.Reshape((image_shape[0], image_shape[1], 1))(x)
    
    
    input_image = keras.Input(shape=(image_shape))
    x = keras.layers.Concatenate()([input_image, x])
    
    
    x = keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    
    x = keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    
    x = keras.layers.Conv2D(filters=128, kernel_size=3, padding='same')(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(x)
    x = keras.layers.LeakyReLU()(x)
    
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(1)(x)
    output = keras.layers.Activation('sigmoid', dtype=tf.float32, name='predictions')(x)
    
    model = keras.Model([input_image, input_label], output, name='discriminator')
    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5), metrics=['accuracy'])
    return model


def get_generator(lat_dim, num_out_channels=3, num_classes=10):
    input_label = keras.Input(shape=(1,))
    x = keras.layers.Embedding(num_classes, 100)(input_label)
    shape = 8 * 8
    x = keras.layers.Dense(shape)(x)
    out_lab = keras.layers.Reshape((8, 8, 1))(x)
    
    input_lat = keras.layers.Input(shape=(lat_dim,))
    x = keras.layers.Dense(128 * shape)(input_lat)
    x = keras.layers.LeakyReLU(0.2)(x)
    x = keras.layers.Reshape((8, 8, 128))(x)
    x = keras.layers.Concatenate()([x, out_lab])
    
    x = keras.layers.Conv2D(256, kernel_size=3, padding='same')(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    x = keras.layers.SpatialDropout2D(0.2)(x)
    
    x = keras.layers.UpSampling2D()(x)
    x = keras.layers.Conv2D(256, kernel_size=3, padding='same')(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    x = keras.layers.SpatialDropout2D(0.2)(x)
    
    x = keras.layers.UpSampling2D()(x)
    x = keras.layers.Conv2D(128, kernel_size=3, padding='same')(x)
    x = keras.layers.LeakyReLU(0.2)(x)

    x = keras.layers.Conv2D(num_out_channels, kernel_size=1, padding='same')(x)
    output = keras.layers.Activation('tanh', dtype=tf.float32, name='predictions')(x)
    
    
    model = keras.Model([input_lat, input_label], output, name='generator')
    return model


def get_gan(g_model: keras.Model, d_model: keras.Model)-> keras.Model:
    d_model.trainable = False
    
    gen_lat, gen_labels = g_model.input
    gen_output = g_model.output
    
    gan_output = d_model([gen_output, gen_labels])
    
    gan_model = keras.Model(inputs=[gen_lat, gen_labels], outputs=gan_output)
    
    gan_model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=5e-4, beta_1=0.5))
    return gan_model

