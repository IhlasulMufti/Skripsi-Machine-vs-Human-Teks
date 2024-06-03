import os
import tensorflow as tf
import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, TFT5EncoderModel

# Function to create checkboxes and return selected models
def get_selected_models():
    selected_models = []
    
    cols = st.columns(3)  # Create 3 columns for checkboxes
    for i, model in enumerate(model_names):
        with cols[i % 3]:  # Distribute checkboxes across the columns
            if st.checkbox(model, key=model):
                selected_models.append(model)
    return selected_models

def plot_metric_accuracy(df, metric_name, selected_models):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(df) + 1)  # Create a range starting from 1 to the number of epochs
    
    for model in selected_models:
        if model in df.columns:
            plt.plot(epochs, df[model], label=model)
        
    plt.title(f'{metric_name} over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.xticks(epochs)  # Set x-axis labels to start from 1
    plt.legend(loc='upper right')
    plt.grid(True)
    st.pyplot(plt)

# Function to plot bar chart for a given metric
def plot_metric_f1score(scores, metric_name):
    sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    models, values = zip(*sorted_models)
    
    # Assign colors: different color for top 3, another color for the rest
    colors = ['gold' if i < 3 else 'skyblue' for i in range(len(models))]
    
    plt.figure(figsize=(10, 6))
    plt.bar(models, values, color=colors)
    plt.xlabel('Models')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} of Different Models')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(plt)

# Function to load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('saved-model/t5_fullMerg.h5', custom_objects={'TFT5EncoderModel': TFT5EncoderModel})
    tokenizer = AutoTokenizer.from_pretrained("michelecafagna26/t5-base-finetuned-sst2-sentiment")  # Adjust the model name as necessary
    return model, tokenizer

def preprocess_text(dataframe, tokenizer, max_length=256):
    inputs = tokenizer(dataframe, padding=True, truncation=True, return_tensors="tf", max_length=max_length)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    return {"input_ids": input_ids, "attention_mask": attention_mask}

# Function to make predictions
def make_prediction(input_text, tokenizer):
    inputs = preprocess_text(input_text, tokenizer)
    predictions = model(inputs)
    # Convert tensor to numpy array and extract the value
    prediction_value = predictions.numpy().flatten()[0]
    return prediction_value

# Custom CSS for styling the button
button_css = """
    <style>
        .stButton > button {
            background-color: #4CAF50; /* Green */
            color: white;
            border: none;
            padding: 10px 28px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 18px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 20px;
            transition: color 0.3s ease; /* Smooth transition for text color */
        }
        .stButton > button:hover {
            color: black; /* Black text on hover */
        }
        .stButton {
            display: flex;
            justify-content: center;
        }
    </style>
"""

# Custom CSS for the result box
result_box_css = """
    <style>
        .result-box {
            background-color: #333; /* Dark grey border */
            color: white; /* White text */
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
        }
    </style>
"""

# Apply the custom CSS
st.markdown(button_css, unsafe_allow_html=True)
st.markdown(result_box_css, unsafe_allow_html=True)

# Sidebar setup for navigation
with st.sidebar:
    model_names = [
        'ernie_abstract', 't5_abstract', 'xlnet_abstract',
        'ernie_wiki',     't5_wiki',     'xlnet_wiki',
        'ernie_balMerg',  't5_balMerg',  'xlnet_balMerg',
        'ernie_fullMerg', 't5_fullMerg', 'xlnet_fullMerg'
    ]

    # Load the combined CSV files
    loss_df = pd.read_csv('combined-history/loss_result.csv')
    accuracy_df = pd.read_csv('combined-history/accuracy_result.csv')
    val_loss_df = pd.read_csv('combined-history/val_loss_result.csv')
    val_accuracy_df = pd.read_csv('combined-history/val_accuracy_result.csv')
    
    page = st.selectbox(
        label="Navigation",
        options=("Analysis Result", "Predict GPT")
    )

if page == "Analysis Result":
    with st.container():
        st.header("Select Models to Review")
        selected_models = get_selected_models()
        
        tab1, tab2 = st.tabs(["Accuracy Score", "F1-Score"])
        
        with tab1:
            if selected_models:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Loss")
                    plot_metric_accuracy(loss_df, 'Loss', selected_models)
                    
                    st.header("Validation Loss")
                    plot_metric_accuracy(val_loss_df, 'Validation Loss', selected_models)
                
                with col2:
                    st.header("Accuracy")
                    plot_metric_accuracy(accuracy_df, 'Accuracy', selected_models)

                    st.header("Validation Accuracy")
                    plot_metric_accuracy(val_accuracy_df, 'Validation Accuracy', selected_models)
            else:
                st.markdown(f'<div class="result-box">Please select at least one model to review the metrics.</div>', unsafe_allow_html=True)
 
        with tab2:
            if selected_models:
                # Dictionaries to store model name and corresponding scores
                precision_scores = {}
                recall_scores = {}
                f1_scores = {}
                
                # Read the scores from each file
                for model_name in selected_models:
                    file_name = f"f1score-history/{model_name}.txt"  # update the file path
                    with open(file_name, 'r') as file:
                        content = file.read()
                        # Extract the scores using regex
                        precision = float(re.search(r'Precision:\s*([\d.]+)', content).group(1))
                        recall = float(re.search(r'Recall:\s*([\d.]+)', content).group(1))
                        f1_score = float(re.search(r'F1 Score:\s*([\d.]+)', content).group(1))
                        
                        precision_scores[model_name] = precision
                        recall_scores[model_name] = recall
                        f1_scores[model_name] = f1_score
                        
                # Plotting Precision, Recall, and F1 Score
                st.header("Precision")
                plot_metric_f1score(precision_scores, 'Precision')
                
                st.header("Recall")
                plot_metric_f1score(recall_scores, 'Recall')
                
                st.header("F1 Score")
                plot_metric_f1score(f1_scores, 'F1 Score')
                        
            else:
                st.markdown(f'<div class="result-box">Please select at least one model to review the metrics.</div>', unsafe_allow_html=True)

elif page == "Predict GPT":
    with st.container():
        # Load the model
        model, tokenizer = load_model()
        
        # Streamlit app interface
        st.title('Text Prediction')
        input_text = st.text_area('Enter text to predict')

        if st.button('Predict'):
            word_count = len(input_text.split())
            if word_count >= 183:
                prediction_value = make_prediction(input_text, tokenizer)
                prediction_percentage = prediction_value * 100
                st.markdown(f'<div class="result-box">Generated by Machine {prediction_percentage:.2f}%</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-box">Please add more text. Text is too short</div>', unsafe_allow_html=True)
