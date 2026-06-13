# pip install openai
import openai

YANDEX_CLOUD_FOLDER = "folder name"
YANDEX_CLOUD_API_KEY = "my api key"
YANDEX_CLOUD_MODEL = "gpt-oss-20b/latest"

client = openai.OpenAI(
  api_key=YANDEX_CLOUD_API_KEY,
  base_url="https://ai.api.cloud.yandex.net/v1",
  project=YANDEX_CLOUD_FOLDER
)

response = client.responses.create(
  model=f"gpt://{YANDEX_CLOUD_FOLDER}/{YANDEX_CLOUD_MODEL}",
  temperature=0.3,
  instructions="",
  input="Answer in one word: works?",
  max_output_tokens=500
)

print(response.output_text)
