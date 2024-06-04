import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer, TFT5EncoderModel

st.set_page_config(page_title="Predict GPT", page_icon="🔮")

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


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        'saved-model/t5_fullMerg.h5', custom_objects={'TFT5EncoderModel': TFT5EncoderModel})
    tokenizer = AutoTokenizer.from_pretrained(
        "michelecafagna26/t5-base-finetuned-sst2-sentiment")
    return model, tokenizer


def preprocess_text(dataframe, tokenizer, max_length=256):
    tokenizer.padding_side = 'left'

    inputs = tokenizer(
        dataframe,
        padding='max_length',
        truncation=True,
        return_tensors="tf",
        max_length=max_length
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def make_prediction(input_text, tokenizer):
    inputs = preprocess_text(input_text, tokenizer)
    predictions = model(inputs)
    prediction_value = predictions.numpy().flatten()[0]
    return prediction_value

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

model, tokenizer = load_model()

st.markdown('# PREDIKSI TEKS GPT')

st.markdown("### Petunjuk Penggunaan")
st.write("""
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse euismod ipsum sem, vitae fermentum est vulputate id. Curabitur venenatis sapien dolor, sed sagittis nulla bibendum eu. Sed condimentum neque et tortor rutrum, quis convallis leo porta. Nam efficitur magna nec turpis commodo hendrerit. Ut sit amet sapien sed ex cursus blandit. Aliquam metus metus, auctor consequat egestas dignissim, dictum sed elit. Duis fermentum ultricies urna in eleifend.
    Morbi dapibus velit augue, a facilisis mi sollicitudin eu. Nam eu hendrerit ligula, vitae ultricies odio. Phasellus at placerat mi, quis vulputate nibh. Sed orci lectus, fermentum et tincidunt sed, dignissim eu purus. Mauris eget lacinia nulla. Donec dapibus, odio ut gravida placerat, ex risus cursus mi, vitae iaculis ligula velit eget orci. Integer ultrices sapien ligula, quis ultrices justo pretium ac. Proin at nunc ullamcorper, scelerisque lectus et, pellentesque sem. Fusce urna quam, malesuada vel nulla sed, fermentum volutpat nibh. In vel lorem dolor. Nulla suscipit diam nulla, vitae rhoncus mi convallis vitae. Phasellus molestie diam eget metus hendrerit, sit amet fringilla eros convallis.       
""")

input_text = st.text_area('Masukkan teks untuk diprediksi')

if st.button('Periksa'):
    word_count = len(input_text.split())

    if word_count >= 100:
        with st.spinner('Wait For Prediction 😉'):
            prediction_value = make_prediction(input_text, tokenizer)

        prediction_percentage = prediction_value * 100
        st.markdown(
            f'<div class="result-box">Generated by Machine {prediction_percentage:.2f}%</div>',
            unsafe_allow_html=True)
    else:
        st.error('Silakan tambahkan lebih banyak teks. Teks terlalu pendek')