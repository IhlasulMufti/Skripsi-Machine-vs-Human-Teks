import streamlit as st
import matplotlib.pyplot as plt


def sidebar():
    with st.sidebar:
        st.markdown(f'<div style= height:160px </div>', unsafe_allow_html=True)
        st.markdown("""
            <div class="logo">
                <img src="https://github.com/IhlasulMufti/Skripsi-Machine-vs-Human-Teks/blob/main/app/assets/lab-sisfor.png?raw=true" width="220">
            </div>
        """, unsafe_allow_html=True)
        st.markdown(f'<div style= height:50px </div>', unsafe_allow_html=True)
        st.divider()
        st.caption('2024 © Ihlasul Mufti Faqih.')


def header_logo():
    st.markdown("""
        <div class="logo">
            <img src="https://github.com/IhlasulMufti/Skripsi-Machine-vs-Human-Teks/blob/main/app/assets/Heading.png?raw=true" width="250">
        </div>
    """, unsafe_allow_html=True)
    st.markdown(f'<div style= height:80px </div>', unsafe_allow_html=True)
    st.divider()


def select_models(model_names):
    selected_models = []

    with st.container():
        st.markdown("#### Pilih model untuk diulas")
        cols = st.columns(3)  # Create 3 columns for checkboxes
        for i, model in enumerate(model_names):
            with cols[i % 3]:  # Distribute checkboxes across the columns
                if st.checkbox(model, key=model):
                    selected_models.append(model)
    return selected_models


def metric_accuracy(df, metric_name, selected_models):
    plt.figure(figsize=(8, 6))
    # Create a range starting from 1 to the number of epochs
    epochs = range(1, len(df) + 1)

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


def metric_f1score(scores, metric_name):
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


def real_time_word_count(text):
    words = text.split()
    return len(words)
