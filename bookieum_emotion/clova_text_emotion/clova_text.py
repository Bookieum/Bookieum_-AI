import requests
import json

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
                
                text_sentiment_score= confidence_positive*0.7 + confidence_negative*0.3

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

client_id = "YOUR_ID"
client_secret = "YOUR_SECRET"

print("오늘, 당신의 하루는 어땠나요? (800자 이내로 적어보세요) ")
sentence = input("답변: ")
result = analyze_sentiment(sentence, client_id, client_secret)
print(json.dumps(result, indent=4, sort_keys=True))
