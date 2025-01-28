from transformers import pipeline, MarianTokenizer, TFMarianMTModel
import streamlit as st

# Загрузка модели для перевода
@st.cache_resource  # Используем st.cache_resource для кэширования моделей
def load_translation_model(src_lang, trg_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{trg_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = TFMarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# Функция для перевода текста
def translate_text(text, tokenizer, model):
    batch = tokenizer([text], return_tensors="tf")
    gen = model.generate(**batch)
    translated_text = tokenizer.batch_decode(gen, skip_special_tokens=True)
    return translated_text[0]

# Загрузка модели для анализа тональности
@st.cache_resource  # Используем st.cache_resource для кэширования моделей
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")

# Заголовок приложения
st.title('Тональность текста и перевод')

# Инициализация моделей
src_lang = "en"  # Исходный язык
trg_lang = "ru"  # Целевой язык
tokenizer, translation_model = load_translation_model(src_lang, trg_lang)
sentiment_model = load_sentiment_model()

# Пример текста для перевода
sample_text = "hello"
translated_sample = translate_text(sample_text, tokenizer, translation_model)
st.write(f"Пример перевода ('{sample_text}' -> русский): {translated_sample}")

# Ввод текста пользователем
input_text = st.text_input('Введите текст на английском:', 'I love Python')

# Кнопка для анализа тональности
if st.button('Определить тональность текста'):
    # Перевод текста
    translated_text = translate_text(input_text, tokenizer, translation_model)
    st.write(f"Перевод: {translated_text}")

    # Анализ тональности
    sentiment_result = sentiment_model(translated_text)[0]
    st.write(f"Тональность: {sentiment_result['label']} (вероятность: {sentiment_result['score']:.2f})")