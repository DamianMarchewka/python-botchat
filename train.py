import json
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def open_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

data = open_json("intents.json")

texts, labels = [], []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(intent["intent"])

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2)
    )),
    ("clf", LogisticRegression(max_iter=1000))
])

pipeline.fit(texts, labels)

joblib.dump(pipeline, "model.joblib")

# zapis odpowiedzi per intencja, żeby bot mógł po nie sięgnąć
responses = {item["intent"]: item["responses"] for item in data["intents"]}
with open("response.json", "w", encoding="utf-8") as f:
    json.dump(responses, f, ensure_ascii=False, indent=2)

print("✅ Model zapisany do model.joblib, odpowiedzi do responses.json")