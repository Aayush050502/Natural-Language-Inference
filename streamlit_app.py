import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

class REACTAgent:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_response(self, sentence1, sentence2):
        inputs = self.tokenizer(sentence1, sentence2, return_tensors='pt', truncation=True, padding='max_length')
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax(dim=-1).item()
        return predicted_class

agent = REACTAgent(model, tokenizer)

st.title('Human-in-the-Loop Agent for NLI')

# HTML and CSS for the custom styling
html_code = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Human-in-the-Loop Agent for NLI</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f0f0f0;
            margin: 0;
            padding: 20px;
        }
        .container {
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            max-width: 600px;
            margin: 40px auto;
        }
        h1 {
            color: #333;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #555;
        }
        input[type="text"] {
            width: calc(100% - 20px);
            padding: 10px;
            margin-bottom: 20px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            padding: 10px 20px;
            background-color: #007bff;
            color: #fff;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #0056b3;
        }
        .result {
            margin-top: 20px;
            font-size: 18px;
            color: #333;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Human-in-the-Loop Agent for NLI</h1>
        <label for="sentence1">Enter the first sentence:</label>
        <input type="text" id="sentence1" name="sentence1">
        <label for="sentence2">Enter the second sentence:</label>
        <input type="text" id="sentence2" name="sentence2">
        <button onclick="getPrediction()">Submit</button>
        <div class="result" id="result"></div>
    </div>
    <script>
        async function getPrediction() {
            const sentence1 = document.getElementById('sentence1').value;
            const sentence2 = document.getElementById('sentence2').value;

            if (sentence1 && sentence2) {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ sentence1, sentence2 })
                });
                const data = await response.json();
                const labels = ['Entailment', 'Contradiction', 'Neutral'];
                document.getElementById('result').innerText = `Predicted label: ${labels[data.label]}`;
            } else {
                document.getElementById('result').innerText = 'Please enter both sentences.';
            }
        }
    </script>
</body>
</html>
"""

# Embed HTML, CSS, and JavaScript in Streamlit app
st.components.v1.html(html_code, height=600)

# API endpoint to handle prediction requests
def predict(sentence1, sentence2):
    label = agent.generate_response(sentence1, sentence2)
    return label

# Streamlit server configuration to handle the /predict endpoint
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def handle_predict():
    data = request.json
    sentence1 = data.get('sentence1')
    sentence2 = data.get('sentence2')
    label = predict(sentence1, sentence2)
    return jsonify({'label': label})

# Run the Flask app if this is the main module
if __name__ == '__main__':
    app.run()
