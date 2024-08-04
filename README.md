Human-in-the-Loop NLI Agent
This project builds a Human-in-the-Loop Agent for Natural Language Inference (NLI) using the Stanford Natural Language Inference (SNLI) dataset. The goal is to create an autonomous workflow that allows for human interaction during task execution. This implementation utilizes REACT (Reasoning and Acting) to support human inputs and interactions during the inference process.

Project Overview
The SNLI corpus (version 1.0) is a collection of 570k human-written English sentence pairs, manually labeled for balanced classification into three categories: entailment, contradiction, and neutral. This dataset supports the task of natural language inference (NLI), also known as recognizing textual entailment (RTE). The dataset is intended to serve as both a benchmark for evaluating text representation systems and as a resource for developing NLP models.

This project aims to leverage this dataset to build a custom LLM (Large Language Model) Human-in-the-Loop Agent. The implementation is done using the BERT model and involves:

Data Preprocessing: Loading, cleaning, and preparing the dataset for training.
Model Training: Fine-tuning a pre-trained BERT model on the SNLI dataset.
REACT Implementation: Creating an interactive agent that can classify sentence pairs with human-in-the-loop capabilities.
Deployment with Streamlit: Making the model accessible via a user-friendly interface using Streamlit.
Acknowledgements
This dataset was kindly made available by the Stanford Natural Language Processing Group.

Installation
To run this project locally, follow these steps:

Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/human-in-the-loop-nli-agent.git
cd human-in-the-loop-nli-agent
Install the required libraries:

bash
Copy code
pip install -r requirements.txt
Download the SNLI dataset:

Download the dataset from here.
Extract the files and place them in a directory named snli_dataset within your project folder.
Usage
To run the Streamlit app:

Create the Streamlit app file:
Save the following content in a file named streamlit_app.py:

python
Copy code
import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Load the dataset
train_file = 'snli_dataset/snli_1.0_train.csv'
valid_file = 'snli_dataset/snli_1.0_dev.csv'
test_file = 'snli_dataset/snli_1.0_test.csv'

df_train = pd.read_csv(train_file, delimiter=',', header=None)
df_valid = pd.read_csv(valid_file, delimiter=',', header=None)
df_test = pd.read_csv(test_file, delimiter=',', header=None)

# Rename columns
df_train.columns = ['sentence1', 'sentence2', 'label']
df_valid.columns = ['sentence1', 'sentence2', 'label']
df_test.columns = ['sentence1', 'sentence2', 'label']

# Preprocess text
def preprocess_text(text):
    return text.lower()

df_train['sentence1'] = df_train['sentence1'].apply(preprocess_text)
df_train['sentence2'] = df_train['sentence2'].apply(preprocess_text)
df_valid['sentence1'] = df_valid['sentence1'].apply(preprocess_text)
df_valid['sentence2'] = df_valid['sentence2'].apply(preprocess_text)
df_test['sentence1'] = df_test['sentence1'].apply(preprocess_text)
df_test['sentence2'] = df_test['sentence2'].apply(preprocess_text)

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length')

train_dataset = Dataset.from_pandas(df_train[['sentence1', 'sentence2', 'label']])
valid_dataset = Dataset.from_pandas(df_valid[['sentence1', 'sentence2', 'label']])
test_dataset = Dataset.from_pandas(df_test[['sentence1', 'sentence2', 'label']])

train_dataset = train_dataset.map(tokenize_function, batched=True)
valid_dataset = valid_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='/content/results',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir='/content/logs',
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

# Train the model
trainer.train()

# Define REACT Agent
class REACTAgent:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_response(self, sentence1, sentence2):
        inputs = self.tokenizer(sentence1, sentence2, return_tensors='pt', truncation=True, padding='max_length')
        outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax(dim=-1).item()
        return predicted_class

# Instantiate REACT Agent
agent = REACTAgent(model, tokenizer)

# Streamlit App
st.title("Human-in-the-Loop NLI Agent")

st.write("### Training Data")
st.dataframe(df_train.head())

st.write("### Validation Data")
st.dataframe(df_valid.head())

st.write("### Test Data")
st.dataframe(df_test.head())

# Define interaction function
def interact_with_agent(sentence1, sentence2):
    label = agent.generate_response(sentence1, sentence2)
    labels = ['Entailment', 'Contradiction', 'Neutral']
    return labels[label]

st.write("### Interact with the Model")

sentence1 = st.text_input("Enter Sentence 1:")
sentence2 = st.text_input("Enter Sentence 2:")

if st.button("Classify"):
    if sentence1 and sentence2:
        result = interact_with_agent(sentence1, sentence2)
        st.write(f"Classification Result: {result}")
    else:
        st.write("Please enter both sentences.")
Run the Streamlit app:

python
Copy code
!pip install streamlit pyngrok
from pyngrok import ngrok

# Terminate any existing ngrok connections
ngrok.kill()

# Create a new ngrok tunnel
public_url = ngrok.connect(port='8501')
print(f"Streamlit URL: {public_url}")

# Run the Streamlit app
!streamlit run streamlit_app.py
This will display a public URL where you can access your Streamlit app.

Conclusion
This project demonstrates how to build and deploy a Human-in-the-Loop NLI Agent using the SNLI dataset and a BERT model. By leveraging Streamlit, we make the model interactive and accessible, allowing for human inputs during the inference process.

Feel free to explore the code and make improvements as needed. Contributions are welcome!
