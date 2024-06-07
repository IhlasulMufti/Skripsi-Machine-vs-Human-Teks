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
            <img src="https://github.com/IhlasulMufti/Skripsi-Machine-vs-Human-Teks/blob/main/asset/lab-sisfor.png?raw=true" width="220">
        </div>
    """, unsafe_allow_html=True)
    st.markdown(f'<div style= height:50px </div>', unsafe_allow_html=True)
    st.divider()
    st.caption('© 2024 Ihlasul Mufti Faqih.')
    
st.markdown("""
    <div class="logo">
        <img src="https://github.com/IhlasulMufti/Skripsi-Machine-vs-Human-Teks/blob/main/asset/Heading.png?raw=true" width="250">
    </div>
""", unsafe_allow_html=True)
st.markdown(f'<div style= height:80px </div>', unsafe_allow_html=True)
st.divider()

st.write("""
    Aplikasi ini bertujuan
""")

st.markdown("### Datasets")
st.write("""
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse euismod ipsum sem, vitae fermentum est vulputate id. Curabitur venenatis sapien dolor, sed sagittis nulla bibendum eu. Sed condimentum neque et tortor rutrum, quis convallis leo porta. Nam efficitur magna nec turpis commodo hendrerit. Ut sit amet sapien sed ex cursus blandit. Aliquam metus metus, auctor consequat egestas dignissim, dictum sed elit. Duis fermentum ultricies urna in eleifend.
    Morbi dapibus velit augue, a facilisis mi sollicitudin eu. Nam eu hendrerit ligula, vitae ultricies odio. Phasellus at placerat mi, quis vulputate nibh. Sed orci lectus, fermentum et tincidunt sed, dignissim eu purus. Mauris eget lacinia nulla. Donec dapibus, odio ut gravida placerat, ex risus cursus mi, vitae iaculis ligula velit eget orci. Integer ultrices sapien ligula, quis ultrices justo pretium ac. Proin at nunc ullamcorper, scelerisque lectus et, pellentesque sem. Fusce urna quam, malesuada vel nulla sed, fermentum volutpat nibh. In vel lorem dolor. Nulla suscipit diam nulla, vitae rhoncus mi convallis vitae. Phasellus molestie diam eget metus hendrerit, sit amet fringilla eros convallis.       
""")

st.markdown("### Fitur Aplikasi")
st.write("""
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse euismod ipsum sem, vitae fermentum est vulputate id. Curabitur venenatis sapien dolor, sed sagittis nulla bibendum eu. Sed condimentum neque et tortor rutrum, quis convallis leo porta. Nam efficitur magna nec turpis commodo hendrerit. Ut sit amet sapien sed ex cursus blandit. Aliquam metus metus, auctor consequat egestas dignissim, dictum sed elit. Duis fermentum ultricies urna in eleifend.
    Morbi dapibus velit augue, a facilisis mi sollicitudin eu. Nam eu hendrerit ligula, vitae ultricies odio. Phasellus at placerat mi, quis vulputate nibh. Sed orci lectus, fermentum et tincidunt sed, dignissim eu purus. Mauris eget lacinia nulla. Donec dapibus, odio ut gravida placerat, ex risus cursus mi, vitae iaculis ligula velit eget orci. Integer ultrices sapien ligula, quis ultrices justo pretium ac. Proin at nunc ullamcorper, scelerisque lectus et, pellentesque sem. Fusce urna quam, malesuada vel nulla sed, fermentum volutpat nibh. In vel lorem dolor. Nulla suscipit diam nulla, vitae rhoncus mi convallis vitae. Phasellus molestie diam eget metus hendrerit, sit amet fringilla eros convallis.       
""")