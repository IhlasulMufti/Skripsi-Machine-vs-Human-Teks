import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Function to create checkboxes and return selected models
def get_selected_models():
    selected_models = []
    
    cols = st.columns(3)  # Create 3 columns for checkboxes
    for i, model in enumerate(model_names):
        with cols[i % 3]:  # Distribute checkboxes across the columns
            if st.checkbox(model, key=model):
                selected_models.append(model)
    return selected_models

def plot_metric(df, metric_name, selected_models):
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
    
    st.header("Navigation")
    page = st.radio("Go to", ["Analysis Result", "Predict GPT"])
    
if page == "Analysis Result":
    st.header("Select Models to Review")
    selected_models = get_selected_models()
    
    tab1, tab2 = st.tabs(["Accuracy Score", "F1-Score"])
    
    with tab1:
        col1, col2 = st.columns(2)
        if selected_models:
            with col1:
                st.header("Loss")
                plot_metric(loss_df, 'Loss', selected_models)
                
                st.header("Validation Loss")
                plot_metric(val_loss_df, 'Validation Loss', selected_models)
            
            with col2:
                st.header("Accuracy")
                plot_metric(accuracy_df, 'Accuracy', selected_models)

                st.header("Validation Accuracy")
                plot_metric(val_accuracy_df, 'Validation Accuracy', selected_models)
        else:
            st.write("Please select at least one model to review the metrics.")
            
    with tab2:
        st.write("Content for Predict tab f1-score")

elif page == "Predict GPT":
    st.write("Content for Predict GPT tab")
