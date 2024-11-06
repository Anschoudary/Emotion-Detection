import kagglehub
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_dataset():
    # Download latest version
    path = kagglehub.dataset_download("msambare/fer2013")
    return path

def get_data_generators():
    # Define the path to your dataset
    data_dir = get_dataset()  # Replace with the actual path to your dataset

    # Define image size and batch size
    img_width, img_height = 48, 48  # Standard size for FER2013
    batch_size = 32

    # Create data generators for training and validation sets
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Normalize pixel values
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        data_dir + '/train/',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode='grayscale',  # Use grayscale for FER2013
        class_mode='categorical'
    )

    validation_generator = test_datagen.flow_from_directory(
        data_dir + '/test/',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical'
    )

    return train_generator, validation_generator