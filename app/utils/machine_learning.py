import streamlit as st

from transformers import AutoTokenizer
from huggingface_hub import from_pretrained_keras


@st.cache_resource
def load_model():
    model = from_pretrained_keras("ihlasulmufti/machine-vs-human-predict-gpt")
    tokenizer = AutoTokenizer.from_pretrained(
        "michelecafagna26/t5-base-finetuned-sst2-sentiment")
    return model, tokenizer


def preprocess_text(text, tokenizer, max_length=256):
    tokenizer.padding_side = 'left'

    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        return_tensors="tf",
        max_length=max_length
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def make_prediction(input_text, model, tokenizer):
    inputs = preprocess_text(input_text, tokenizer)
    predictions = model(inputs)
    prediction_value = predictions.numpy().flatten()[0]
    return prediction_value
