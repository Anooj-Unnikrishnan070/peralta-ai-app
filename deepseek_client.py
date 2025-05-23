# deepseek_client.py
# import subprocess, json

# class DeepSeekClient:
#     def query(self, prompt: str) -> dict:
#         res = subprocess.run(
#             ["ollama", "run", "deepseek-coder:6.7b"],
#             input=prompt,
#             capture_output=True,
#             text=True,
#             timeout=60
#         )
#         return json.loads(res.stdout)

# deepseek_client = DeepSeekClient()

# ----------------------------------------------------------------------------

# deepseek_client.py
import openai
import streamlit as st

# Configure DeepSeek API
openai.api_key = st.secrets["DEEPSEEK_API_KEY"]
openai.api_base = "https://api.deepseek.com"

class DeepSeekClient:
    def query(self, prompt: str) -> str:
        response = openai.ChatCompletion.create(
            model="deepseek-chat",  # use "deepseek-coder" if needed
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message["content"]

deepseek_client = DeepSeekClient()

