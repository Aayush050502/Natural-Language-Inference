import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
except Exception as e:
    st.error(f"Error loading model or tokenizer: {e}")

class REACTAgent:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_response(self, sentence1, sentence2):
        try:
            inputs = self.tokenizer(sentence1, sentence2, return_tensors='pt', truncation=True, padding='max_length')
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class = logits.argmax(dim=-1).item()
            return predicted_class
        except Exception as e:
            st.error(f"Error generating response: {e}")
            return None

agent = REACTAgent(model, tokenizer)

st.title('Human-in-the-Loop Agent for NLI')

sentence1 = st.text_input("Enter the first sentence:")
sentence2 = st.text_input("Enter the second sentence:")

if sentence1 and sentence2:
    label = agent.generate_response(sentence1, sentence2)
    if label is not None:
        labels = ['Entailment', 'Contradiction', 'Neutral']
        st.write(f"Predicted label: {labels[label]}")
