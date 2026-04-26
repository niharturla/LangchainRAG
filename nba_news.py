import requests
import csv

url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/news"
response = requests.get(url)
data = response.json()
articles = data['articles']

fields = ["id", "type", "headline", "description", "published", "lastModified", "byline"]

with open('news.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(fields)
    for item in articles:
        writer.writerow(
            [
                item.get("id", ""),
                item.get("type", ""),
                item.get("headline", ""),
                item.get("description", ""),
                item.get("published", ""),
                item.get("lastModified", ""),
                item.get("byline", ""),
            ]
        )
print("news.csv has been updated.")