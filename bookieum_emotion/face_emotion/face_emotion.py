import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

def load_emotion_model(model_path):
    return load_model(model_path)

def detect_emotions(video_capture, emotion_model):
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    emotion_counts = {label: 0 for label in emotion_labels}

    while True:
        _, frame = video_capture.read()
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (224, 224), interpolation=cv2.INTER_AREA)
            roi = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
            roi = roi.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=0)

            prediction = emotion_model.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            emotion_counts[label] += 1

        cv2.imshow('Emotion Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    return emotion_counts

def calculate_emotion_scores(emotion_counts):
    positive_emotions = ['Happy', 'Neutral', 'Surprise']
    negative_emotions = ['Angry', 'Disgust', 'Fear', 'Sad']

    positive_score = sum(emotion_counts[label] for label in positive_emotions)
    negative_score = sum(emotion_counts[label] for label in negative_emotions)

    total_emotions = sum(emotion_counts.values())
    positive_normalized = round(positive_score / total_emotions, 7)
    negative_normalized = round(negative_score / total_emotions, 7)
    face_sentiment_score= round(positive_normalized *0.7 + negative_normalized*0.3,3)
    return positive_normalized, negative_normalized,face_sentiment_score

# Usage example:
face_classifier = cv2.CascadeClassifier(r"C:\Users\user\Desktop\hackathon\bookieum_emotion\face_emotion\haarcascade_frontalface_default.xml")
emotion_model = load_emotion_model(r"C:\Users\user\Desktop\hackathon\bookieum_emotion\face_emotion\Emotion_Detection.h5")
cap = cv2.VideoCapture(0)

emotion_counts = detect_emotions(cap, emotion_model)
positive_score, negative_score, face_sentiment_score = calculate_emotion_scores(emotion_counts)
print(f"Positive Score: {positive_score:.3f}")
print(f"Negative Score: {negative_score:.3f}")
print(f"face_sentiment_score: {face_sentiment_score:.3f}")
