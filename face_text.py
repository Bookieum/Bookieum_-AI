import requests
import json
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import threading
import multiprocessing
import concurrent.futures
from multiprocessing import Process, Queue
import tensorflow as tf
import os
import keyboard

# 텍스트 감정 분석 코드
def analyze_sentiment(sentence, client_id, client_secret):
    url = "https://naveropenapi.apigw.ntruss.com/sentiment-analysis/v1/analyze"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": client_id,
        "X-NCP-APIGW-API-KEY": client_secret,
        "Content-Type": "application/json"
    }

    data = {
        "content": sentence
    }

    response = requests.post(url, data=json.dumps(data), headers=headers)
    rescode = response.status_code

    if rescode == 200:
        result = json.loads(response.text)
        document = result.get("document")
        if document:
            sentiment = document.get("sentiment")
            confidence = document.get("confidence")
            if sentiment and confidence:
                confidence_neutral = confidence.get("neutral") / 100.0 / 2.0
                confidence_positive = confidence.get("positive") / 100.0 + confidence_neutral
                confidence_negative = confidence.get("negative") / 100.0 + confidence_neutral

                text_sentiment_score = confidence_positive * 0.7 + confidence_negative * 0.3

                return {
                    "Sentiment": sentiment,
                    "Positive Confidence": round(confidence_positive, 7),
                    "Negative Confidence": round(confidence_negative, 7),
                    "Text Sentiment Score": round(text_sentiment_score, 3)
                }
            else:
                return {"error": "Sentiment and Confidence information not found"}
        else:
            return {"error": "Document information not found"}
    else:
        return {"error": "HTTP Error: " + response.text}


# 얼굴 감정 분석 코드
def load_emotion_model(model_path):
    return load_model(model_path)


def detect_emotions(video_capture, emotion_model, face_classifier):
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

            with tf.device('/CPU:0'):
                prediction = emotion_model.predict(roi, verbose=0)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            emotion_counts[label] += 1

        #cv2.imshow('Emotion Detector', frame)

        # if cv2.waitKey(1)  == ord('q'):
        #     break
        if keyboard.is_pressed('enter'):
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
    face_sentiment_score = round(positive_normalized * 0.7 + negative_normalized * 0.3, 3)
    return positive_normalized, negative_normalized, face_sentiment_score


def face_analysis_thread(video_capture, emotion_model,face_classifier,result_queue):
    emotion_counts = detect_emotions(video_capture, emotion_model, face_classifier)
    positive_score, negative_score, face_sentiment_score = calculate_emotion_scores(emotion_counts)
    # print(f"Face Analysis - Positive Score: {positive_score:.3f}")
    # print(f"Face Analysis - Negative Score: {negative_score:.3f}")
    # print(f"Face Analysis - Face Sentiment Score: {face_sentiment_score:.3f}")
    result_queue.put({"face_sentiment_score": face_sentiment_score})


def text_analysis_thread(result_queue):
    client_id = "hpgzp4aq0t"
    client_secret = "frhqiMUgx0HT3j7AtF9fJ9h3w3hqm2w9bX7YRjK5"

    print("오늘, 당신의 하루는 어땠나요? (800자 이내로 적어보세요) ")
    sentence = input("답변: ")
    result = analyze_sentiment(sentence, client_id, client_secret)
    # print("Text Analysis - Results:")
    # print(json.dumps(result, indent=4, sort_keys=True))
    result_queue.put({"text_sentiment_score": result["Text Sentiment Score"]})


# if __name__ == "__main__":
def main():
    # Load face classifier and emotion model
    face_classifier = cv2.CascadeClassifier(r"C:\Users\USER\hackathon\bookieum_emotion\face_emotion\haarcascade_frontalface_default.xml")
    emotion_model = load_emotion_model(r"C:\Users\USER\hackathon\bookieum_emotion\face_emotion\Emotion_Detection.h5")

    # Open video capture
    cap = cv2.VideoCapture(0)
    result_queue = Queue()
    # Create threads for face analysis and text analysis
    face_thread = threading.Thread(target=face_analysis_thread, args=(cap, emotion_model,face_classifier,result_queue))
    text_thread = threading.Thread(target=text_analysis_thread, args=(result_queue,))

    # Start both threads
    face_thread.start()
    text_thread.start()

    # Wait for both threads to finish
    face_thread.join()
    text_thread.join()

    # Release video capture
    cap.release()
    cv2.destroyAllWindows()

    # Retrieve results from the Queue
    results = {"face_sentiment_score": None, "text_sentiment_score": None}
    while not result_queue.empty():
        result = result_queue.get()
        results.update(result)

    # Calculate average sentiment scores
    average_sentiment = (results["face_sentiment_score"] + results["text_sentiment_score"]) / 2.0

    # Print the results
    print(f"Average Sentiment Score: {average_sentiment:.3f}")
    return average_sentiment
if __name__ == "__main__":
    main()