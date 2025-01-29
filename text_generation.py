from huggingface_hub import InferenceClient
import streamlit as st

client = InferenceClient(
	provider="hf-inference",
	api_key="HugginFace_Token"
)



messages = [
	{
		"role": "user",
		"content": "What is the capital of France?"
	}
]

completion = client.chat.completions.create(
    model="mistralai/Mistral-Nemo-Instruct-2407", 
	messages=messages, 
	max_tokens=500
)


st.title('Генерация текста')
st.write(f"Текст: {сompletion.choices[0].message}")
