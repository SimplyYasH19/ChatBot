from flask import Flask, request, jsonify, render_template
import json
import random
import pickle
import numpy as np
import requests
import urllib.parse
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load model and preprocessing objects
model = keras.models.load_model("chat_model.keras")

with open("tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pickle", "rb") as f:
    lbl_encoder = pickle.load(f)

with open("intents.json") as file:
    data = json.load(file)

max_len = 20


# Wikipedia helper functions

def clean_query(query):
    remove_list = [
        "what is ", "tell me about ", "who is ",
        "explain ", "give me info about ", "can you explain ",
        "what are ", "what's ", "define "
    ]

    q = query.lower()
    for word in remove_list:
        q = q.replace(word, "")

    return q.strip()


def fetch_wikipedia_summary(user_query):
    try:
        clean_q = clean_query(user_query)
        encoded_query = urllib.parse.quote(clean_q)

        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_query}"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})

        if response.status_code == 200:
            data = response.json()
            return data.get("extract", None)
        else:
            return None

    except:
        return None

def is_topic_in_intents(user_input):
    text = user_input.lower()

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            if pattern.lower() in text or text in pattern.lower():
                return True

    return False


# Core chatbot logic

def get_response(user_input):
    user_input_clean = user_input.lower().strip()

    # If topic is NOT in intents → go to Wikipedia directly
    if not is_topic_in_intents(user_input_clean):
        summary = fetch_wikipedia_summary(user_input)
        if summary:
            return summary
        else:
            return "Sorry, I am not sure about that."

    # Otherwise, use ML model
    seq = pad_sequences(
        tokenizer.texts_to_sequences([user_input]),
        truncating="post",
        maxlen=max_len
    )

    result = model.predict(seq)
    predicted_tag = lbl_encoder.inverse_transform([np.argmax(result)])[0]

    for intent in data["intents"]:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])

    return "Sorry, I am not sure about that."


# Flask routes

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]
    bot_reply = get_response(user_message)
    return jsonify({"reply": bot_reply})


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
