# Emoticare Machine Learning Documentation
Simple Chatbot about Mental Health using TensorFlow in Python.

## Tools and Python Modules Used
- Kaggle Notebooks
- PyCharm Community Edition 2024.1.1
- Visual Studio Code

## Data Collection
This project involves the development of a simple mental health chatbot model by leveraging three distinct datasets. Here's a detailed breakdown of each dataset:
#### Dataset 1: Chatbot for Mental Health Conversations
The [Chatbot for Mental Health Conversations](https://www.kaggle.com/code/jocelyndumlao/chatbot-for-mental-health-conversations/) is obtained from Kaggle.
#### Dataset 2: Therapist Patient Conversation Dataset
The [Therapist Patient Conversation Dataset](https://www.kaggle.com/datasets/neelghoshal/therapist-patient-conversation-dataset) is obtained from Kaggle.
#### Dataset 3: Conversations dataset for chatbot
The [Conversations dataset for chatbot](https://www.kaggle.com/datasets/kanikamalhotra1307/conversations-dataset-for-chatbot) is obtained from Kaggle.

The three datasets were merged into a single master dataset using the code provided in `dataset-merged.ipynb`. The merging process was based on intents, which include tag, patterns, and responses. Duplicate entries were then removed from the dataset. After merging, the dataset was saved as combined_dataset.json and then renamed to `intense.json`. The purpose of merging the datasets was to enrich the dataset with relevant information. 

## Training the Model
The code in 'training.py' handles the training process for the chatbot model. Here's a breakdown of the key steps involved:
#### 1. Text Preprocessing
The script utilizes a class called WordNetLemmatizer() to perform word lemmatization. This process identifies the base or root form of words, ensuring the chatbot can recognize various inflections of the same word. 
#### 2. One-Hot Encoding
This step prepares the text data for training a neural network. It converts words into a format of 1s and 0s by creating a "bag" that indicates whether a word exists in a specific pattern. The data is then shuffled, converted to a numerical array, and split into training data (train_x) and training labels (train_y).
#### 3. Model Building with Sequential Model
The script utilizes a Sequential model, a common architecture for building neural networks. This model is trained on the prepared dataset, allowing it to learn the patterns and relationships between user inputs and desired responses. By analyzing these patterns, the model develops the ability to generate appropriate responses to user queries during live interactions.

This process will generate three output files: `chatbotmodel.h5`, `classes.pkl`, and `words.pkl`.

## Run the Chatbot
After the model training is complete, the `main.py` code is then created to test the chatbot model's ability to respond to given inputs.
