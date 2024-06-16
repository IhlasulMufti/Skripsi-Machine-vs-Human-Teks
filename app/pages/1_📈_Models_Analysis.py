import streamlit as st
import pandas as pd
import re

from utils.display import sidebar, header_logo, select_models, metric_accuracy, metric_f1score


st.set_page_config(
    page_title="Models Analysis",
    page_icon="📈"
)


with open("app/css/styles.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

with st.container():
    sidebar()

with st.container():
    header_logo()

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

    st.divider()

    selected_models = select_models(model_names)

tab1, tab2 = st.tabs(["Accuracy Score", "Confusion Matrix"])
with tab1:
    loss_df = pd.read_csv('app/data/combined-history/loss_result.csv')
    accuracy_df = pd.read_csv('app/data/combined-history/accuracy_result.csv')
    val_loss_df = pd.read_csv('app/data/combined-history/val_loss_result.csv')
    val_accuracy_df = pd.read_csv(
        'app/data/combined-history/val_accuracy_result.csv')

    with st.expander("Tabel"):
        st.write("""
                Tingkat akurasi model dapat dilihat pada tabel. Tingkat akurasi tertinggi ditandai dengan warna kuning.
                Jika dilihat secara keseluruhan model yang dilatih menggunakan dataset wikipedia menjadi dataset dengan akurasi
                tertinggi, hanya kalah pada arsitektur XLNet.
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
                Tingkat akurasi model dapat dilihat pada tabel. Tingkat akurasi tertinggi ditandai dengan warna kuning.
                Jika dilihat secara keseluruhan model yang dilatih menggunakan dataset wikipedia menjadi dataset dengan akurasi
                tertinggi, hanya kalah pada arsitektur XLNet.
            """)

    if selected_models:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Loss")
            metric_accuracy(loss_df, 'Loss', selected_models)

            st.markdown("### Accuracy")
            metric_accuracy(
                accuracy_df, 'Accuracy', selected_models)

        with col2:
            st.markdown("### Validation Loss")
            metric_accuracy(
                val_loss_df, 'Validation Loss', selected_models)

            st.markdown("### Validation Accuracy")
            metric_accuracy(
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
            Tingkat akurasi model dapat dilihat pada tabel. Tingkat akurasi tertinggi ditandai dengan warna kuning.
            Jika dilihat secara keseluruhan model yang dilatih menggunakan dataset wikipedia menjadi dataset dengan akurasi
            tertinggi, hanya kalah pada arsitektur XLNet.
        """)

    if selected_models:
        for model_name in selected_models:
            file_name = f"app/data/f1score-history/{model_name}.txt"
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
        metric_f1score(precision_scores, 'Precision')

        st.markdown("### Recall")
        metric_f1score(recall_scores, 'Recall')

        st.markdown("### F1 Score")
        metric_f1score(f1_scores, 'F1 Score')
    else:
        st.warning(
            "Harap pilih setidaknya satu model untuk meninjau metriknya.")
