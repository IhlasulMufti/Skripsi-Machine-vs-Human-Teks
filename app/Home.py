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


st.write("""
    Aplikasi ini bertujuan
""")

st.markdown("### Datasets")
st.write("""
    Tingkat akurasi model dapat dilihat pada tabel. Tingkat akurasi tertinggi ditandai dengan warna kuning.
    Jika dilihat secara keseluruhan model yang dilatih menggunakan dataset wikipedia menjadi dataset dengan akurasi
    tertinggi, hanya kalah pada arsitektur XLNet.
""")

st.markdown("### Fitur Aplikasi")
st.write("""
    Terdapat dua halaman yang dapat diakses.
    1. Model Analysis - Halaman ini berisikan semua hal yang berkaitan dengan 
""")
