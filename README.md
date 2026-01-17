# 🤖 AI Chatbot Using NLP and Machine Learning

This project is an AI-based chatbot that can interact with users using natural language. The chatbot is trained using a machine learning model to classify user messages into predefined intents such as greeting, help, company information, and general queries.

If the user asks a question that is not present in the training dataset, the chatbot automatically fetches relevant information from Wikipedia and displays it as the response.

The application runs locally on a laptop using a Flask web server and provides a simple web-based chat interface.

---

## 🖥️ Application Preview

<img width="1724" height="946" alt="Screenshot 2026-01-17 130656" src="https://github.com/user-attachments/assets/d77a38e6-07ce-4d82-87b0-ce7faf3a8288" />


---

## 📌 Project Overview

The goal of this project is to demonstrate an end-to-end NLP-based chatbot system, starting from intent dataset creation and preprocessing to model training and real-time interaction through a web application.

The focus is on:

- Applying NLP techniques for text processing  
- Training a neural network for intent classification  
- Combining ML-based responses with online knowledge (Wikipedia)  
- Converting a console-based chatbot into a web application  

---

## ✨ Features

- Web-based chat interface using HTML, CSS, and JavaScript  
- Intent-based responses using a trained ML model  
- Supports greetings, help, jokes, and company info (Google, SKF, etc.)  
- Automatically fetches answers from Wikipedia for unknown questions  
- Runs completely on a local machine  
- Can be exposed to the internet temporarily using ngrok for demo  

---

## 📚 Dataset

The project uses a custom JSON-based intents dataset (`intents.json`) containing:

- Intent tags  
- Example user patterns  
- Predefined responses  

---

## 🧰 Tech Stack

- Python  
- TensorFlow / Keras  
- Flask  
- HTML, CSS, JavaScript  
- NumPy  
- Scikit-learn  
- Wikipedia REST API  
- ngrok  

---

## ⚙️ How It Works

1. User types a message in the web interface  
2. The Flask backend receives the message  
3. The message is converted into tokens and padded  
4. The trained neural network predicts the intent  
5. If the intent is known, a predefined response is returned  
6. If the intent is unknown, the chatbot fetches information from Wikipedia  
7. The response is sent back to the web interface and displayed to the user  

