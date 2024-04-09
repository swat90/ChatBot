import os
import numpy as np
import pandas as pd

import pickle
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import torch



class QLearningChatbot:
    def __init__(self, states, learning_rate=0.9, discount_factor=0.1):
        self.states = states
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_values = np.random.rand(len(states))
        self.mood = "Neutral"
        self.mood_history = [] 
        self.mood_history_int = []
        self.tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        self.bert_sentiment_model_path = "bert_sentiment.pkl"
        self.bert_sentiment_model = self.load_model() if os.path.exists(self.bert_sentiment_model_path) else AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")


    def detect_sentiment(self, input_text):
        # Encode the text
        encoded_input = self.tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512)
        
        # Perform inference
        with torch.no_grad():
            output = self.bert_sentiment_model(**encoded_input)
        
        # Process the output (softmax to get probabilities)
        scores = softmax(output.logits, dim=1)
        
        # Map scores to sentiment labels
        labels = ['Negative', 'Moderately Negative', 'Neutral', 'Moderately Positive', 'Positive']
        scores = scores.numpy().flatten()
        scores_dict = {label: score for label, score in zip(labels, scores)}
        highest_sentiment = max(scores_dict, key=scores_dict.get)
        self.mood = highest_sentiment
        return highest_sentiment

    def update_q_values(self, current_state, reward, next_state):
        # print(f"state-reward: {current_state} - {reward} -- (b)")
        current_state_index = self.states.index(current_state)
        next_state_index = self.states.index(next_state)

        current_q_value = self.q_values[current_state_index]
        max_next_q_value = np.max(self.q_values[next_state_index])

        new_q_value = current_q_value + self.learning_rate * (reward + self.discount_factor * max_next_q_value - current_q_value)
        self.q_values[current_state_index] = new_q_value

    def update_mood_history(self):
        st.session_state.entered_mood.append(self.mood)
        self.mood_history = st.session_state.entered_mood
        return self.mood_history

    def check_mood_trend(self):
        mood_dict = {'Negative': 1, 'Moderately Negative': 2, 'Neutral': 3, 'Moderately Positive': 4, 'Positive': 5}

        if len(self.mood_history) >= 2:
            self.mood_history_int = [mood_dict.get(x) for x in self.mood_history]
            recent_moods = self.mood_history_int[-2:]
            if recent_moods[-1] > recent_moods[-2]:
                return 'increased'
            elif recent_moods[-1] < recent_moods[-2]:
                return 'decreased'
            else:
                return 'unchanged'
        else:
            return 'unchanged'