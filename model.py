#-----------------------GENERAL IMPORTS----------------------
import os
#USED FOR GETTING DATASET AND EXTRACTING DATA FROM ZIP FOLDER
import urllib.request
import zipfile

#USED FOR GPU RUNNING OF TENSORFLOW
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#GENERAL TENSORFLOW IMPORTS
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

#-----------------------MAIN FUNCTION FOR MODEL TRAINING----------------------

def solution_model():

    #DATASET COLLECTION AND EXTRACTION IN TEMPORARY LOCATION
    '''url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
    urllib.request.urlretrieve(url, 'rps.zip')
    local_zip = 'rps.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/')
    zip_ref.close()
    '''

    #IMAGE DATA GENERATOR INITIALISATION
    TRAINING_DIR = "tmp/rps/"
    train_datagen = ImageDataGenerator(
        rescale=1 / 255,
        rotation_range=40,
        width_shift_range=.2,
        height_shift_range=.2,
        shear_range=.2,
        zoom_range=.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    #TRAINING DATA GENERATOR
    train_generator = train_datagen.flow_from_directory(
        TRAINING_DIR,
        batch_size=64,
        class_mode='categorical',
        target_size=(150, 150)
    )

    #MODEL DEFINITION - LAYERS
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (2, 2), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    '''
    #MODEL2 DEFINITION - TRANSFER LEARNING
    import os
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras import Model
    from tensorflow.keras.applications.inception_v3 import InceptionV3

    // https://drive.google.com/file/d/19ZAfxBrOYEsUSZunqIp6LsPDR2jcbDNJ/view?usp=sharing
    import wget
    #downloading zip folder using wget
    url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
    filename = wget.download(url)
    print(filename)
    

    # Create an instance of the inception model from the local pre-trained weights
    local_weights_file = 'C:/Users/Tarun/Desktop/comp/TFExam/rock-paper-scissors/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

    pre_trained_model = InceptionV3(
        input_shape=(150, 150, 3),
        include_top=False,
        weights=None
    )

    pre_trained_model.load_weights(local_weights_file)
    for layer in pre_trained_model.layers:
        layer.trainable = False

    # Print the model summary
    pre_trained_model.summary()

    last_layer = pre_trained_model.get_layer('mixed7')
    print('last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output

    from tensorflow.keras.optimizers import RMSprop

    # Flatten the output layer to 1 dimension
    x = layers.Flatten()(last_output)
    # Add a fully connected layer with 1,024 hidden units and ReLU activation
    x = layers.Dense(1024, activation='relu')(x)
    # Add a dropout rate of 0.2
    x = layers.Dropout(.2)(x)
    # Add a final sigmoid layer for classification
    x = layers.Dense(1, activation='sigmoid')(x)

    model = Model(pre_trained_model.input, x)
    '''

    #MODEL COMPILATION
    from tensorflow.keras.optimizers import RMSprop
    model.compile(
        optimizer=RMSprop(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    #MODEL TRAINING
    model.fit(train_generator,epochs=20)
    return model

#-----------------------RUN----------------------
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
