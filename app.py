import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import tensorflow as tf
from transformers import AutoTokenizer, TFT5EncoderModel

# Custom CSS for improved styling and positioning the logo
st.markdown("""
    <style>
        body {
            font-family: 'Roboto', sans-serif;
        }
        .main {
            background-color: #1e1e2f;
            color: white;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 28px;
            text-align: center;
            font-size: 18px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 20px;
        }
        .stButton > button:hover {
            color: black;
        }
        .stButton {
            display: flex;
            justify-content: center;
        }
        .result-box {
            background-color: #333;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
        }
        .logo {
            position: absolute;
            top: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

# Function to create checkboxes and return selected models
def get_selected_models():
    selected_models = []
    
    with st.container():
        st.write("Select Models to Review")
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
    plt.legend(loc='upper left')  # Change legend location to top left
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

def plot_metric_f1score(scores, metric_name):
    # Sort the models based on their scores in descending order
    sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    models, values = zip(*sorted_models)
    
    # Define colors for the bars
    colors = ['gold' if i < 3 else 'skyblue' for i in range(len(models))]
    
    # Create the bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, values, color=colors)
    
    # Add text labels to the bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}', 
                 ha='center', va='bottom')
    
    # Set chart labels and title
    plt.xlabel('Models')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} of Different Models')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(plt)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('saved-model/t5_fullMerg.h5', custom_objects={'TFT5EncoderModel': TFT5EncoderModel})
    tokenizer = AutoTokenizer.from_pretrained("michelecafagna26/t5-base-finetuned-sst2-sentiment")
    return model, tokenizer

def preprocess_text(dataframe, tokenizer, max_length=256):
    inputs = tokenizer(dataframe, padding=True, truncation=True, return_tensors="tf", max_length=max_length)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    return {"input_ids": input_ids, "attention_mask": attention_mask}

def make_prediction(input_text, tokenizer):
    inputs = preprocess_text(input_text, tokenizer)
    predictions = model(inputs)
    prediction_value = predictions.numpy().flatten()[0]
    return prediction_value

model_names = [
    'ernie_abstract', 't5_abstract', 'xlnet_abstract',
    'ernie_wiki', 't5_wiki', 'xlnet_wiki',
    'ernie_balMerg', 't5_balMerg', 'xlnet_balMerg',
    'ernie_fullMerg', 't5_fullMerg', 'xlnet_fullMerg'
]

loss_df = pd.read_csv('combined-history/loss_result.csv')
accuracy_df = pd.read_csv('combined-history/accuracy_result.csv')
val_loss_df = pd.read_csv('combined-history/val_loss_result.csv')
val_accuracy_df = pd.read_csv('combined-history/val_accuracy_result.csv')

# Sidebar setup for navigation
with st.sidebar:            
    with st.container():
        # Add your logo image using HTML
        st.markdown(
            """
            <div class="logo">
                <img src="https://github.com/IhlasulMufti/Skripsi-Machine-vs-Human-Teks/blob/main/asset/lab-sisfor.png?raw=true" width="220">
            </div>
            """, unsafe_allow_html=True
        )
    st.markdown(f'<div style= height:50px </div>', unsafe_allow_html=True)
    st.divider()
    
    with st.container():
        page = st.selectbox(
            label="Navigation",
            options=("Home", "Analysis Result", "Predict GPT")
        )
        
    st.markdown(f'<div style= height:215px </div>', unsafe_allow_html=True)    
    st.caption('© 2024 Ihlasul Mufti Faqih.')
    
match page:
    case "Home":
        with st.container():
            # Add your logo image using HTML
            st.markdown(
                """
                <div class="logo">
                    <img src="https://github.com/IhlasulMufti/Skripsi-Machine-vs-Human-Teks/blob/main/asset/Heading.png?raw=true" width="250">
                </div>
                """, unsafe_allow_html=True
            )
        st.markdown(f'<div style= height:80px </div>', unsafe_allow_html=True)
        st.divider()
        
        st.header("Home")
    
    case "Analysis Result":
        with st.container():
            # Add your logo image using HTML
            st.markdown(
                """
                <div class="logo">
                    <img src="https://github.com/IhlasulMufti/Skripsi-Machine-vs-Human-Teks/blob/main/asset/Heading.png?raw=true" width="250">
                </div>
                """, unsafe_allow_html=True
            )
        st.markdown(f'<div style= height:80px </div>', unsafe_allow_html=True)
        st.divider()
            
        with st.container():
            st.header("Analysis Results")
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
                st.warning("Please select at least one model to review the metrics.")
                
        with tab2:
            if selected_models:
                precision_scores = {}
                recall_scores = {}
                f1_scores = {}
                
                for model_name in selected_models:
                    file_name = f"f1score-history/{model_name}.txt"
                    with open(file_name, 'r') as file:
                        content = file.read()
                        precision = float(re.search(r'Precision:\s*([\d.]+)', content).group(1))
                        recall = float(re.search(r'Recall:\s*([\d.]+)', content).group(1))
                        f1_score = float(re.search(r'F1 Score:\s*([\d.]+)', content).group(1))
                        
                        precision_scores[model_name] = precision
                        recall_scores[model_name] = recall
                        f1_scores[model_name] = f1_score
                        
                st.header("Precision")
                plot_metric_f1score(precision_scores, 'Precision')
                
                st.header("Recall")
                plot_metric_f1score(recall_scores, 'Recall')
                
                st.header("F1 Score")
                plot_metric_f1score(f1_scores, 'F1 Score')
            else:
                st.warning("Please select at least one model to review the metrics.")

    case "Predict GPT":
        model, tokenizer = load_model()
        
        with st.container():
            # Add your logo image using HTML
            st.markdown(
                """
                <div class="logo">
                    <img src="https://github.com/IhlasulMufti/Skripsi-Machine-vs-Human-Teks/blob/main/asset/Heading.png?raw=true" width="250">
                </div>
                """, unsafe_allow_html=True
            )
        st.markdown(f'<div style= height:80px </div>', unsafe_allow_html=True)
        st.divider()
        
        st.header('Text Prediction')
        input_text = st.text_area('Enter text to predict')

        
        if st.button('Predict'):
            word_count = len(input_text.split())
            if word_count >= 183:
                with st.spinner('Wait For Prediction 😉'):
                    prediction_value = make_prediction(input_text, tokenizer)
                prediction_percentage = prediction_value * 100
                st.markdown(f'<div class="result-box">Generated by Machine {prediction_percentage:.2f}%</div>', unsafe_allow_html=True)
            else:
                st.error('Please add more text. Text is to short')
