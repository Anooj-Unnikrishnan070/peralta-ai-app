
# The below code is a client for DeepSeek that uses the `ollama` command line tool to run the model locally. 
# deepseek_client.py

import subprocess, json

class DeepSeekClient:
    def query(self, prompt: str) -> dict:
        res = subprocess.run(
            ["ollama", "run", "deepseek-coder:6.7b"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=60
        )
        # return json.loads(res.stdout)
        return res.stdout.strip()

deepseek_client = DeepSeekClient()

# ----------------------------------------------------------------------------
# # While using DeepSeek API, you use the below code instead of the subprocess call.
# # deepseek_client.py
# ----------------------------------------------------------------------------
# # Uncomment the following lines if you want to use the DeepSeek API instead of the local model.
# ----------------------------------------------------------------------------

# from openai import OpenAI
# import streamlit as st

# # Setup client
# client = OpenAI(
#     api_key=st.secrets["DEEPSEEK_API_KEY"],
#     base_url="https://api.deepseek.com"
# )

# class DeepSeekClient:
#     def query(self, prompt: str) -> str:
#         response = client.chat.completions.create(
#             model="deepseek-chat",
#             messages=[
#                 {"role": "user", "content": prompt}
#             ]
#         )
#         return response.choices[0].message.content.strip()

# deepseek_client = DeepSeekClient()

