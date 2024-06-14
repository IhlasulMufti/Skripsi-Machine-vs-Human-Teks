import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="👋",
)

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
    </style>
""", unsafe_allow_html=True)

# Sidebar setup for navigation
with st.sidebar:
    st.markdown(f'<div style= height:160px </div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="logo">
            <img src="https://github.com/IhlasulMufti/Skripsi-Machine-vs-Human-Teks/blob/main/app/assets/lab-sisfor.png?raw=true" width="220">
        </div>
    """, unsafe_allow_html=True)
    st.markdown(f'<div style= height:50px </div>', unsafe_allow_html=True)
    st.divider()
    st.caption('© 2024 Ihlasul Mufti Faqih.')
    
st.markdown("""
    <div class="logo">
        <img src="https://github.com/IhlasulMufti/Skripsi-Machine-vs-Human-Teks/blob/main/app/assets/Heading.png?raw=true" width="250">
    </div>
""", unsafe_allow_html=True)
st.markdown(f'<div style= height:80px </div>', unsafe_allow_html=True)
st.divider()

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