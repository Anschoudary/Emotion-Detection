from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def Model():
    # Build the CNN model
    modal = Sequential()
    modal.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
    modal.add(MaxPooling2D((2, 2)))
    modal.add(Conv2D(64, (3, 3), activation='relu'))
    modal.add(MaxPooling2D((2, 2)))
    modal.add(Conv2D(128, (3, 3), activation='relu'))
    modal.add(MaxPooling2D((2, 2)))
    modal.add(Flatten())
    modal.add(Dense(128, activation='relu'))
    modal.add(Dropout(0.5))
    modal.add(Dense(7, activation='softmax'))

    # Compile the model
    modal.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return modal
