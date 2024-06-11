import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re

st.set_page_config(page_title="Models Analysis", page_icon="📈")

# Custom CSS
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
        table {
            width: 100%;
            border-collapse: collapse;
            color: white;
            font-size: 13px;
        }
        th, td {
            border: 1px solid #1e1e2f;
            text-align: left;
            padding: 8px;
            font-size: 13px;
        }
        th {
            background-color: #1C1C1C;
            color: white;
            font-size: 16px;
        }
        tr:nth-child(even) {
            background-color: #6A6A6A;
        }
        tr:nth-child(odd) {
            background-color: #414141;
        }
        a {
            color: #0013FF; /* DodgerBlue color for links */
            text-decoration: none; /* Remove underline */
        }
        a:hover {
            text-decoration: underline; /* Underline on hover */
        }
    </style>
""", unsafe_allow_html=True)

def get_selected_models():
    selected_models = []

    with st.container():
        st.markdown("#### Pilih model untuk diulas")
        cols = st.columns(3)  # Create 3 columns for checkboxes
        for i, model in enumerate(model_names):
            with cols[i % 3]:  # Distribute checkboxes across the columns
                if st.checkbox(model, key=model):
                    selected_models.append(model)
    return selected_models

def plot_metric_accuracy(df, metric_name, selected_models):
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

# Sidebar setup for navigation
with st.sidebar:
    st.markdown(f'<div style= height:160px </div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="logo">
            <img src="https://github.com/IhlasulMufti/Skripsi-Machine-vs-Human-Teks/blob/main/asset/lab-sisfor.png?raw=true" width="220">
        </div>
    """, unsafe_allow_html=True)
    st.markdown(f'<div style= height:50px </div>', unsafe_allow_html=True)
    st.divider()
    st.caption('© 2024 Ihlasul Mufti Faqih.')
    
# Unhas Logo
with st.container():
    st.markdown(
        """
        <div class="logo">
            <img src="https://github.com/IhlasulMufti/Skripsi-Machine-vs-Human-Teks/blob/main/asset/Heading.png?raw=true" width="250">
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown(f'<div style= height:80px </div>',
                unsafe_allow_html=True)
    st.divider()

model_names = [
    'ernie_abstract', 't5_abstract', 'xlnet_abstract',
    'ernie_wiki', 't5_wiki', 'xlnet_wiki',
    'ernie_balMerg', 't5_balMerg', 'xlnet_balMerg',
    'ernie_fullMerg', 't5_fullMerg', 'xlnet_fullMerg'
]

with st.container():
    st.markdown("# HASIL MODEL TRAINING")
    st.markdown("""
            Semua nama model yang ada pada penelitian ini merupakan campuran dari nama arsitektur dan dataset yang digunakan.
            Contohnya: ernie_abstract, berarti arsitektur yang digunakan adalah ernie dan ditraining menggunakan dataset abstract.
    """)

    with st.expander("Detail Model"):
        st.html("""
        <table>
            <tr>
                <th>Dataset</th>
                <th>Keterangan</th>
            </tr>
            <tr>
                <td>Abstract</td>
                <td>Dataset ini berisikan teks yang diambil dari berbagai abstrak penelitian dengan jumlah 7.000 data untuk
                    masing-masing teks hasil mesin/kecerdasan buatan dan teks hasil manusia.</td>
            </tr>
            <tr>
                <td>Wiki</td>
                <td>Dataset ini berisikan teks yang diambil dari berbagai situs wikipedia dengan jumlah 72.000 data untuk
                    masing-masing teks hasil mesin/kecerdasan buatan dan teks hasil manusia.</td>
            </tr>
            <tr>
                <td>BalMerg</td>
                <td>Dataset ini berisikan penggabungan datasets abstrak dan wikipedia dengan jumlah data seimbang (14.000) 
                    untuk masing-masing abstrak dan wikipedia.</td>
            </tr>
            <tr>
                <td>FullMerg</td>
                <td>Dataset ini berisikan penggabungan datasets abstrak dan wikipedia dengan jumlah data keseluruhan (14.000 
                    abstrak dan 144.000) abstrak dan wikipedia.</td>
            </tr>
            <tr>
                <th>Model</th>
                <th>Keterangan</th>
            </tr>
            <tr>
                <td>ERNIE</td>
                <td>ERNIE merupakan pre-trained model berdasarkan large-scale textual corpora atau kumpulan data teks yang sangat 
                    besar disertai dengan knowledge graph, model ini dapat mengambil leksikal (makna/pemakaian kata), sintaksis
                    (struktur/susunan kalimat), dan informasi pengetahuan dari teks secara bersamaan. Lebih lanjut silahkan 
                    <a href="https://aclanthology.org/P19-1139/" style="color: #E71414;">lihat</a> publikasi.</td>
            </tr>
            <tr>
                <td>XLNet</td>
                <td>XLNet adalah metode pre-training generalisasi auto-regressive (AR) yang menggabungkan keunggulan dari kedua pendekatan
                    auto-regresif (AR) dan autoencoding (AE) melalui objektif pemodelan bahasa permutasi. Arsitektur neural network XLNet
                    dirancang secara cermat untuk berkolaborasi dengan objektif AR, dengan mengintegrasikan elemen seperti Transformer-XL
                    dan mekanisme twostream yang dirancang dengan baik.
                    <a href="https://arxiv.org/abs/1906.08237" style="color: #E71414;">lihat</a> publikasi.</td>
            </tr>
            <tr>
                <td>T5</td>
                <td> Text-to-Text Transfer Transformation (T5) menerapkan metode text-to-text dimana model akan mengambil teks
                    sebagai input dan memproduksi teks baru sebagai output. Kerangka kerja text-to-text memungkinkan untuk
                    secara langsung menerapkan model, tujuan, prosedur training, dan proses decoding yang sama untuk setiap
                    tugas. Lebih lanjut silahkan 
                    <a href="https://www.jmlr.org/papers/volume21/20-074/20-074.pdf" style="color: #E71414;">lihat</a> publikasi.</td>
            </tr>
        </table>
        """)
    
    st.markdown("### 📖 Petunjuk Penggunaan")
    st.markdown("""
            Untuk memeriksa sebuah teks adalah hasil buatan mesin/kecerdasan buatan atau bukan, silahkan ikuti petunjuk
            berikut:
            1. ✔️ Pilih model yang akan dianalisis perbandingannya.
            2. 📈 Pilih tab "Accuracy Score" untuk melihat nilai akurasi dan akurasi validasi.
            3. 📊 Pilih tab "Confusion Matrix" untuk melihat nilai Precision, Recall, dan F1 Score.
            4. 🔄 Jika gambar gagal muncul silahkan hapus pilihan pada salah satu model kemudian pilih kembali.
    """)
    
    selected_models = get_selected_models()

tab1, tab2 = st.tabs(["Accuracy Score", "Confusion Matrix"])
with tab1:
    loss_df = pd.read_csv('combined-history/loss_result.csv')
    accuracy_df = pd.read_csv('combined-history/accuracy_result.csv')
    val_loss_df = pd.read_csv('combined-history/val_loss_result.csv')
    val_accuracy_df = pd.read_csv(
        'combined-history/val_accuracy_result.csv')
    
    with st.expander("Tabel"):
            st.write("""
                Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse euismod ipsum sem, vitae fermentum est vulputate id. Curabitur venenatis sapien dolor, sed sagittis nulla bibendum eu. Sed condimentum neque et tortor rutrum, quis convallis leo porta. Nam efficitur magna nec turpis commodo hendrerit. Ut sit amet sapien sed ex cursus blandit. Aliquam metus metus, auctor consequat egestas dignissim, dictum sed elit. Duis fermentum ultricies urna in eleifend.
                Morbi dapibus velit augue, a facilisis mi sollicitudin eu. Nam eu hendrerit ligula, vitae ultricies odio. Phasellus at placerat mi, quis vulputate nibh. Sed orci lectus, fermentum et tincidunt sed, dignissim eu purus. Mauris eget lacinia nulla. Donec dapibus, odio ut gravida placerat, ex risus cursus mi, vitae iaculis ligula velit eget orci. Integer ultrices sapien ligula, quis ultrices justo pretium ac. Proin at nunc ullamcorper, scelerisque lectus et, pellentesque sem. Fusce urna quam, malesuada vel nulla sed, fermentum volutpat nibh. In vel lorem dolor. Nulla suscipit diam nulla, vitae rhoncus mi convallis vitae. Phasellus molestie diam eget metus hendrerit, sit amet fringilla eros convallis.       
            """)
            
    if selected_models:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Accuracy")
            st.dataframe(
                accuracy_df[selected_models].style.highlight_max(axis=1))
        with col2:
            st.markdown("### Validation Accuracy")
            st.dataframe(
                val_accuracy_df[selected_models].style.highlight_max(axis=1))
    else:
        st.warning(
            "Harap pilih setidaknya satu model untuk meninjau metriknya.")

    st.divider()
        
    with st.expander("Grafik"):
            st.write("""
                Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse euismod ipsum sem, vitae fermentum est vulputate id. Curabitur venenatis sapien dolor, sed sagittis nulla bibendum eu. Sed condimentum neque et tortor rutrum, quis convallis leo porta. Nam efficitur magna nec turpis commodo hendrerit. Ut sit amet sapien sed ex cursus blandit. Aliquam metus metus, auctor consequat egestas dignissim, dictum sed elit. Duis fermentum ultricies urna in eleifend.
                Morbi dapibus velit augue, a facilisis mi sollicitudin eu. Nam eu hendrerit ligula, vitae ultricies odio. Phasellus at placerat mi, quis vulputate nibh. Sed orci lectus, fermentum et tincidunt sed, dignissim eu purus. Mauris eget lacinia nulla. Donec dapibus, odio ut gravida placerat, ex risus cursus mi, vitae iaculis ligula velit eget orci. Integer ultrices sapien ligula, quis ultrices justo pretium ac. Proin at nunc ullamcorper, scelerisque lectus et, pellentesque sem. Fusce urna quam, malesuada vel nulla sed, fermentum volutpat nibh. In vel lorem dolor. Nulla suscipit diam nulla, vitae rhoncus mi convallis vitae. Phasellus molestie diam eget metus hendrerit, sit amet fringilla eros convallis.       
            """)
            
    if selected_models:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Loss")
            plot_metric_accuracy(loss_df, 'Loss', selected_models)

            st.markdown("### Accuracy")
            plot_metric_accuracy(
                accuracy_df, 'Accuracy', selected_models)

        with col2:
            st.markdown("### Validation Loss")
            plot_metric_accuracy(
                val_loss_df, 'Validation Loss', selected_models)

            st.markdown("### Validation Accuracy")
            plot_metric_accuracy(
                val_accuracy_df, 'Validation Accuracy', selected_models)
    else:
        st.warning(
            "Harap pilih setidaknya satu model untuk meninjau metriknya.")

with tab2:
    precision_scores = {}
    recall_scores = {}
    f1_scores = {}
    
    with st.expander("Penjelasan"):
        st.write("""
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse euismod ipsum sem, vitae fermentum est vulputate id. Curabitur venenatis sapien dolor, sed sagittis nulla bibendum eu. Sed condimentum neque et tortor rutrum, quis convallis leo porta. Nam efficitur magna nec turpis commodo hendrerit. Ut sit amet sapien sed ex cursus blandit. Aliquam metus metus, auctor consequat egestas dignissim, dictum sed elit. Duis fermentum ultricies urna in eleifend.
            Morbi dapibus velit augue, a facilisis mi sollicitudin eu. Nam eu hendrerit ligula, vitae ultricies odio. Phasellus at placerat mi, quis vulputate nibh. Sed orci lectus, fermentum et tincidunt sed, dignissim eu purus. Mauris eget lacinia nulla. Donec dapibus, odio ut gravida placerat, ex risus cursus mi, vitae iaculis ligula velit eget orci. Integer ultrices sapien ligula, quis ultrices justo pretium ac. Proin at nunc ullamcorper, scelerisque lectus et, pellentesque sem. Fusce urna quam, malesuada vel nulla sed, fermentum volutpat nibh. In vel lorem dolor. Nulla suscipit diam nulla, vitae rhoncus mi convallis vitae. Phasellus molestie diam eget metus hendrerit, sit amet fringilla eros convallis.       
        """)

    if selected_models:
        for model_name in selected_models:
            file_name = f"f1score-history/{model_name}.txt"
            with open(file_name, 'r') as file:
                content = file.read()
                precision = float(
                    re.search(r'Precision:\s*([\d.]+)', content).group(1))
                recall = float(
                    re.search(r'Recall:\s*([\d.]+)', content).group(1))
                f1_score = float(
                    re.search(r'F1 Score:\s*([\d.]+)', content).group(1))

                precision_scores[model_name] = precision
                recall_scores[model_name] = recall
                f1_scores[model_name] = f1_score

        st.markdown("### Precision")
        plot_metric_f1score(precision_scores, 'Precision')

        st.markdown("### Recall")
        plot_metric_f1score(recall_scores, 'Recall')

        st.markdown("### F1 Score")
        plot_metric_f1score(f1_scores, 'F1 Score')
    else:
        st.warning(
            "Harap pilih setidaknya satu model untuk meninjau metriknya.")