from transformers import pipeline
import io
import streamlit as st
from transformers import MarianTokenizer, TFMarianMTModel

@st.cache(allow_output_mutation=True)
def translation(str_cl):
    batch = tokenizer([str_cl], return_tensors="tf")
    gen = model.generate(**batch)
    tr=tokenizer.batch_decode(gen, skip_special_tokens=True)
    st.write('Рус:', str(tr))

@st.cache(allow_output_mutation=True)
def model_load(a,b):
    return pipeline(a,b)

classifier = model_load("sentiment-analysis", "blanchefort/rubert-base-cased-sentiment")

st.title('Тональность текста')
src = "en"  # source language
trg = "ru"  # target language
sample_text = "hello"
model_name = f"Helsinki-NLP/opus-mt-{src}-{trg}"
model = TFMarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)
translation(sample_text)

inp_text = st.text_input('Англ:', 'Обожаю питон')
st.write('',inp_text)

result = st.button('Определить тональность текста')

if result:
   st.write('Англ: ',inp_text)
   cl = classifier(str(inp_text))[0]
   st.write(cl)
   #for i in cl:
   #    st.write(str(i["label"]),' с вероятностью ',str(100*float(i["score"])),'%')