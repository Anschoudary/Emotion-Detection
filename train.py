from dataset import get_data_generators
from models import Model

def training():
    # Get the data generators
    train_generator, validation_generator = get_data_generators()

    # Build the model
    model = Model()

    batch_size = 32

    # Train the model using the data generators
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=100,  # Adjust the number of epochs as needed
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )

    print(model.history)

    model.save('emotion_detection_model.h5')

    return model