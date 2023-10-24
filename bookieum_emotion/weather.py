import requests
import json

key = 'YOUR_API_KEY'
send_url = 'http://api.ipstack.com/check?access_key=' + key
r = requests.get(send_url)
j = json.loads(r.text)

# 경도
lon = j['longitude']

# 위도
lat = j['latitude']

apiKey = "YOUR_API_KEY"

api = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={apiKey}"
result=requests.get(api)
result=json.loads(result.text)
i=0
temp=result.copy()
weather=temp['list'][i]['weather'][0]['main']
print("날씨 : ",weather)
print("강수량 : ",temp['list'][i]['rain']['3h'] if temp['list'][i]['weather'][0]['main']=='rain' else 0)
