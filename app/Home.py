import streamlit as st
from utils.display import sidebar, header_logo


st.set_page_config(
    page_title="Home",
    page_icon="👋"
)


with open("app/css/styles.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

with st.container():
    sidebar()

with st.container():
    header_logo()


with st.container():
    st.markdown("""
        <div style="text-align: justify">
            Aplikasi ini dibuat dengan tujuan untuk menampilkan hasil penelitian dan pengaplikasian secara langsung
            hasil penelitian. Penelitian yang dilakukan berjudul <em>ANALISIS PERBANDINGAN KINERJA MODEL ARSITEKTUR XLNET, 
            T5, DAN ERNIE TERHADAP AI-GENERATED TEXT DETECTION</em> dan hasil akhirnya adalah sebuah fitur yang dapat melakukan 
            pendeteksian teks yang berpotensi dibuat menggunakan AI seperti ChatGPT, Gemini, dll.
        </div>
        
        ## Datasets

        <div style="text-align: justify">
            Dataset yang digunakan pada penelitian diambil dari komunitas Huggingface. Dataset terdiri dari teks jenis abstrak penelitian
            dan wikipedia. Untuk melihat lebih lengkapnya silahkan lihat
            <a href="https://huggingface.co/datasets/NicolaiSivesind/human-vs-machine" style="color: #FF0000;">disini</a>.
        </div>

        ## Fitur Aplikasi
        
        Terdapat dua halaman yang dapat diakses dengan cara memilih halaman yang ada pada sidebar.
        1. 📈 Model Analysis __ Halaman ini berisikan semua hal yang berkaitan dengan hasil penelitian.
        2. 🔮 Predict GPT __ Halaman ini berisikan fitur pendeteksian teks yang berpotensi dibuat menggunakan AI *(Artificial Intelligence)*
    """, unsafe_allow_html=True)
