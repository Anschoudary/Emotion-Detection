import cv2
import numpy as np
from train import training

# Function to get frames from the video capture
def get_frame(cap):
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Return the frame
    return frame

# Function to get the emotion from a frame
def get_emotion(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face ROI
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype('float32') / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        # Predict the emotion
        prediction = model.predict(roi_gray)[0]
        emotion_label = emotion_labels[np.argmax(prediction)]

        # Draw a rectangle around the face and display the emotion label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

# Function to display the video stream
def display_video():
    # Start video capture
    cap = cv2.VideoCapture(0)  # Use 0 for default camera

    while True:
        # Get a frame from the video capture
        frame = get_frame(cap)

        # Get the emotion from the frame
        frame = get_emotion(frame)

        # Display the frame
        cv2.imshow('Emotion Detection', frame)

        # Exit if any key is pressed
        if cv2.waitKey(1) & 0xFF != 255:
            break

    # Release the video capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Load the pre-trained model
    model = training()

    # Load the face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Define the emotion labels
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    if input("Do you want to display the video stream? (y/n): ") == 'y':
        # Display the video stream
        display_video()