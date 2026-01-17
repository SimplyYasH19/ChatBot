## AI Chatbot Using NLP and Machine Learning
This project is an AI-based chatbot that can interact with users using natural language. The chatbot is trained using a machine learning model to classify user messages into predefined intents
such as greeting, help, company information, and general queries.
If the user asks a question that is not present in the training dataset, the chatbot automatically fetches relevant information from Wikipedia and displays it as the response.
The application runs locally on a laptop using a Flask web server and provides a simple web-based chat interface.

🖥️ Preview
<img width="1724" height="946" alt="Screenshot 2026-01-17 130656" src="https://github.com/user-attachments/assets/0e89f08b-f3d5-4bc3-aa2b-70d35dbf7b8b" />


📌 Project Overview
The goal of this project is to demonstrate an end-to-end NLP-based chatbot system, starting from intent dataset creation and preprocessing to model training and real-time interaction through a web application.
The focus is on:
    1.Applying NLP techniques for text processing
    2.Training a neural network for intent classification
    3.Combining ML-based responses with online knowledge (Wikipedia)
    4.Converting a console-based chatbot into a web application


✨ Features
    1.Web-based chat interface using HTML, CSS, and JavaScript
    2.Intent-based responses using a trained ML model
    3.Supports greetings, help, company info etc. 
    4.Automatically fetches answers from Wikipedia for unknown questions
    5.Runs completely on a local machine
    6.Can be exposed to the internet temporarily using ngrok for demo


📚 Dataset
  1.The project uses a custom JSON-based intents dataset (intents.json) containing:
  2.Intent tags  
  3.Example user patterns
  4.Predefined responses


🧰 Tech Stack

  Programming Language: Python
  
  Machine Learning: TensorFlow / Keras
  
  NLP: Tokenization, Padding, Label Encoding
  
  Backend: Flask
  
  Frontend: HTML, CSS, JavaScript
  
  Data Storage: JSON, Pickle files
  
  External API: Wikipedia REST API
  
  Deployment (Demo): ngrok


⚙️ How It Works:

  1. User types a message in the web interface.
  2. The Flask backend receives the message.
  3. The message is converted into tokens and padded.  
  4. The trained neural network predicts the intent.
  5. If the intent is known, a predefined response is returned.
  6. If the intent is unknown, the chatbot fetches information from Wikipedia.
  7. The response is sent back to the web interface and displayed to the user.

🚀 Future Improvements

. Add more intents and training data
. Improve intent detection accuracy
. Add user authentication and chat history
. Deploy permanently on cloud (Render / HuggingFace)
. Integrate a more advanced language model
