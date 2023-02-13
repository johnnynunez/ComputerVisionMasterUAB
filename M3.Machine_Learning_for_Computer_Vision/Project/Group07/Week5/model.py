#  from tensorflow.keras.activations import relu, softmax, sigmoid, tanh, elu, selu, softplus, softsign, leaky_relu
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
import tensorflow as tf



def MyModel(name,
            img_dim,
            num_blocks,
            second_layer,
            third_layer,
            num_denses,
            dim_dense,
            filters1,
            filters2,
            batch_norm,
            dropout,
            dropout_range,
            kernel_size,
            kernel_regularizer,
            strides,
            non_linearities,
            initializer,
            pool_size):
    
    # # Define mish using keras backend
    def mish(x):
        return x * K.tanh(K.softplus(x))


    print("Creating model: ", name)
    print("num_blocks: ", num_blocks)
    print("img_dim: ", img_dim)
    print("second_layer: ", second_layer)
    print("third_layer: ", third_layer)
    print("num_denses: ", num_denses)
    print("filters: ", filters1)
    print("batch_norm: ", batch_norm)
    print("dropout: ", dropout)
    print("dropout_range: ", dropout_range)
    print("kernel_size: ", kernel_size)
    print("kernel_regularizer: ", kernel_regularizer)
    print("strides: ", strides)
    print("non_linearities: ", non_linearities)
    print("initializer: ", initializer)
    print("pool_size: ", pool_size)

    if non_linearities == 'mish':
        non_linearities = mish

    if kernel_regularizer == True:
        kernel_regularizer = regularizers.L1L2(l1=1e-5, l2=1e-4)
    else:
        kernel_regularizer = None

    
    if name == "SeqNet":
        # Model BO (JOHNNY)
        Sequential = tf.keras.Sequential()
        # First convolutional block --> input (256, 256, 3) --> output (128, 128, 32)
        Sequential.add(
            tf.keras.layers.Conv2D(filters1[0], kernel_size, strides, padding="same", activation=non_linearities,
                                   kernel_initializer=initializer,
                                   input_shape=(img_dim, img_dim, 3)))
        if second_layer:
            Sequential.add(
                tf.keras.layers.Conv2D(filters1[1], kernel_size, strides, padding="same", activation=non_linearities,
                                       kernel_regularizer=kernel_regularizer,
                                       kernel_initializer=initializer))
        Sequential.add(tf.keras.layers.MaxPool2D(pool_size=pool_size))
        if batch_norm:
            Sequential.add(tf.keras.layers.BatchNormalization())

        if num_blocks > 1:
            # Second convolutional block  --> input (128, 128, 32) --> output (64, 64, 64)
            Sequential.add(
                tf.keras.layers.Conv2D(filters1[2], kernel_size, strides, padding="same", activation=non_linearities,
                                       kernel_regularizer=kernel_regularizer,
                                       kernel_initializer=initializer))
            if second_layer:
                Sequential.add(tf.keras.layers.Conv2D(filters1[3], kernel_size, strides, padding="same",
                                                      activation=non_linearities,
                                                      kernel_regularizer=kernel_regularizer,
                                                      kernel_initializer=initializer))
            Sequential.add(tf.keras.layers.MaxPool2D(pool_size=pool_size))
            if batch_norm:
                Sequential.add(tf.keras.layers.BatchNormalization())

        if num_blocks > 2:
            # Third convolutional block 
            Sequential.add(
                tf.keras.layers.Conv2D(filters1[4], kernel_size, strides, padding="same", activation=non_linearities,
                                       kernel_regularizer=kernel_regularizer,
                                       kernel_initializer=initializer))
            if second_layer:
                Sequential.add(tf.keras.layers.Conv2D(filters1[5], kernel_size, strides, padding="same",
                                                      activation=non_linearities,
                                                      kernel_regularizer=kernel_regularizer,
                                                      kernel_initializer=initializer))
            Sequential.add(tf.keras.layers.MaxPool2D(pool_size=pool_size))
            if batch_norm:
                Sequential.add(tf.keras.layers.BatchNormalization())

        if num_blocks > 3:
            # Fourth convolutional block
            Sequential.add(
                tf.keras.layers.Conv2D(filters1[6], kernel_size, strides, padding="same", activation=non_linearities,
                                       kernel_regularizer=kernel_regularizer,
                                       kernel_initializer=initializer))
            if second_layer:
                Sequential.add(tf.keras.layers.Conv2D(filters1[7], kernel_size, strides, padding="same",
                                                      activation=non_linearities,
                                                      kernel_regularizer=kernel_regularizer,
                                                      kernel_initializer=initializer))
                if third_layer:
                    Sequential.add(tf.keras.layers.Conv2D(filters1[7], kernel_size, strides, padding="same",
                                                          activation=non_linearities,
                                                          kernel_regularizer=kernel_regularizer,
                                                          kernel_initializer=initializer))
            Sequential.add(tf.keras.layers.MaxPool2D(pool_size=pool_size))
            if batch_norm:
                Sequential.add(tf.keras.layers.BatchNormalization())

        if num_blocks > 4:
            # Fourth convolutional block
            Sequential.add(tf.keras.layers.Conv2D(512, kernel_size, strides, padding="same", activation=non_linearities,
                                                  kernel_regularizer=kernel_regularizer,
                                                  kernel_initializer=initializer))
            if second_layer:
                Sequential.add(
                    tf.keras.layers.Conv2D(512, kernel_size, strides, padding="same", activation=non_linearities,
                                           kernel_regularizer=kernel_regularizer,
                                           kernel_initializer=initializer))
            Sequential.add(tf.keras.layers.MaxPool2D(pool_size=pool_size))
            if batch_norm:
                Sequential.add(tf.keras.layers.BatchNormalization())

        Sequential.add(tf.keras.layers.GlobalAveragePooling2D())

        if num_denses == 4:
            Sequential.add(tf.keras.layers.Dense(dim_dense, activation=non_linearities,
                                                 kernel_initializer=initializer, kernel_regularizer=kernel_regularizer))
            if dropout:
                Sequential.add(tf.keras.layers.Dropout(dropout_range + 0.4))
            Sequential.add(tf.keras.layers.Dense(int(dim_dense // 2), activation=non_linearities,
                                                 kernel_initializer=initializer, kernel_regularizer=kernel_regularizer))
            if dropout:
                Sequential.add(tf.keras.layers.Dropout(dropout_range + 0.3))
            Sequential.add(tf.keras.layers.Dense(int(dim_dense // 4), activation=non_linearities,
                                                 kernel_initializer=initializer, kernel_regularizer=kernel_regularizer))
            if dropout:
                Sequential.add(tf.keras.layers.Dropout(dropout_range + 0.2))
            Sequential.add(tf.keras.layers.Dense(int(dim_dense // 8), activation=non_linearities,
                                                 kernel_initializer=initializer, kernel_regularizer=kernel_regularizer))
            if dropout:
                Sequential.add(tf.keras.layers.Dropout(dropout_range))
        if num_denses == 3:
            Sequential.add(tf.keras.layers.Dense(dim_dense, activation=non_linearities,
                                                 kernel_initializer=initializer, kernel_regularizer=kernel_regularizer))
            if dropout:
                Sequential.add(tf.keras.layers.Dropout(dropout_range[0]))
            Sequential.add(tf.keras.layers.Dense(int(dim_dense // 2), activation=non_linearities,
                                                 kernel_initializer=initializer, kernel_regularizer=kernel_regularizer))
            if dropout:
                Sequential.add(tf.keras.layers.Dropout(dropout_range[1]))
            Sequential.add(tf.keras.layers.Dense(int(dim_dense // 8), activation=non_linearities,
                                                 kernel_initializer=initializer, kernel_regularizer=kernel_regularizer))
            if dropout:
                Sequential.add(tf.keras.layers.Dropout(dropout_range[2]))
        if num_denses == 2:
            Sequential.add(tf.keras.layers.Dense(128, activation=non_linearities,
                                                 kernel_initializer=initializer, kernel_regularizer=kernel_regularizer))
            if dropout:
                Sequential.add(tf.keras.layers.Dropout(dropout_range + 0.3))
            Sequential.add(tf.keras.layers.Dense(64, activation=non_linearities,
                                                 kernel_initializer=initializer, kernel_regularizer=kernel_regularizer))
            if dropout:
                Sequential.add(tf.keras.layers.Dropout(dropout_range))
        if num_denses == 1:
            Sequential.add(tf.keras.layers.Dense(64, activation=non_linearities,
                                                 kernel_initializer=initializer, kernel_regularizer=kernel_regularizer))
            if dropout:
                Sequential.add(tf.keras.layers.Dropout(dropout_range))

        Sequential.add(tf.keras.layers.Dense(8, activation='softmax', kernel_initializer=initializer))


    elif name == "ResNet":
        input = tf.keras.Input(shape=(64, 64, 3))
        x = tf.keras.layers.Conv2D(filters1, (3, 3), padding="same", activation=non_linearities,
                                   kernel_initializer=initializer)(input)
        x = tf.keras.layers.Conv2D(filters1, (3, 3), padding="same", activation=non_linearities,
                                   kernel_initializer=initializer)(x)
        # x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        b1_out = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters1, (3, 3), padding="same", activation=non_linearities,
                                   kernel_initializer=initializer)(b1_out)
        x = tf.keras.layers.Conv2D(filters1, (3, 3), padding="same", activation=non_linearities,
                                   kernel_initializer=initializer)(x)
        #  x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        b2_out = tf.keras.layers.BatchNormalization()(x)
        res_1 = tf.keras.layers.Add()([b1_out, b2_out])
        x = tf.keras.layers.Conv2D(filters1, (3, 3), padding="same", activation=non_linearities,
                                   kernel_initializer=initializer)(res_1)
        x = tf.keras.layers.Conv2D(filters1, (3, 3), padding="same", activation=non_linearities,
                                   kernel_initializer=initializer)(x)
        #  x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        b3_out = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([b1_out, b2_out, b3_out])

        if num_blocks == 4:
            x = tf.keras.layers.Conv2D(filters1, (3, 3), padding="same", activation=non_linearities,
                                       kernel_initializer=initializer)(x)
            x = tf.keras.layers.Conv2D(filters1, (3, 3), padding="same", activation=non_linearities,
                                       kernel_initializer=initializer)(x)
            #  x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
            b4_out = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Add()([b1_out, b2_out, b3_out, b4_out])

        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        # Feed the network to the fully connected layers
        if num_denses == 3:
            x = tf.keras.layers.Dense(256, activation=non_linearities, kernel_initializer=initializer)(x)
            x = tf.keras.layers.Dropout(0.6)(x)
            x = tf.keras.layers.Dense(128, activation=non_linearities, kernel_initializer=initializer)(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(64, activation=non_linearities, kernel_initializer=initializer)(x)
            x = tf.keras.layers.Dropout(0.3)(x)
        elif num_denses == 2:
            x = tf.keras.layers.Dense(128, activation=non_linearities, kernel_initializer=initializer)(x)
            x = tf.keras.layers.Dropout(0.6)(x)
            x = tf.keras.layers.Dense(64, activation=non_linearities, kernel_initializer=initializer)(x)
            x = tf.keras.layers.Dropout(0.5)(x)
        elif num_denses == 1:
            x = tf.keras.layers.Dense(64, activation=non_linearities, kernel_initializer=initializer)(x)
            x = tf.keras.layers.Dropout(0.6)(x)
        output = tf.keras.layers.Dense(8, activation='softmax', kernel_initializer=initializer)(x)

        Sequential = tf.keras.Model(inputs=input, outputs=output)


    elif name == "basic":
        Sequential = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=(64, 64, 3), activation='relu'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            # tf.keras.layers.Flatten(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(8, activation='softmax')
        ])

    elif name == "basic2":
        Sequential = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=(64, 64, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            # tf.keras.layers.Flatten(),
            tf.keras.layers.GlobalAveragePooling2D(),
            # tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(8, activation='softmax')
        ])
    elif name == "basic3":
        Sequential = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=(64, 64, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            # tf.keras.layers.Flatten(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(8, activation='softmax')
        ])
    elif name == "basic4":
        Sequential = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=(64, 64, 3), activation='relu'),
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
            tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation='relu'),
            tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(8, activation="softmax")
        ])
    elif name == "basic5":
        Sequential = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=(64, 64, 3), activation='relu'),
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
            tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(8, activation="softmax")
        ])
    
    elif name == "autoencoder":
        encoder_inputs = tf.keras.layers.Input(shape=(img_dim, img_dim, 3))
        x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(encoder_inputs)  # 64 x 64 x 3
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)  # 32 x 32 x 32
        x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)  # 16 x 16 x 64

        x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)  # 16 x 16 x 128
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)  # 8 x 8 x 64

        x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)  # 8 x 8 x 32
        x = tf.keras.layers.BatchNormalization()(x)
        encoder_outputs = x

        x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(encoder_outputs)  # 8 x 8 x 128
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)

        x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)  # 8 x 8 x 128
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)

        x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)  # 8 x 8 x 128
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        decoder_outputs = tf.keras.layers.Dense(8, activation='softmax', kernel_initializer="HeUniform")(x)

        encoder = tf.keras.Model(encoder_inputs, encoder_outputs, name="encoder")
        decoder = tf.keras.Model(encoder_outputs, decoder_outputs, name="decoder")
        Sequential = tf.keras.Model(encoder_inputs, decoder(encoder(encoder_inputs)), name="encoder_decoder")

    return Sequential



