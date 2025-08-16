import json
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

with open("intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

texts, labels = [], []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(intent["intent"])

print("Texts:", texts)
print("Labels:", labels)