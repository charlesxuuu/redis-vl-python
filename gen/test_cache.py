import os
import getpass
import time
import numpy as np

from openai import OpenAI


os.environ["TOKENIZERS_PARALLELISM"] = "False"

# api_key = os.getenv("OPENAI_API_KEY") or getpass.getpass("Enter your OpenAI API key: ")
# Define the file path for the API key
api_key_file = os.path.join(os.path.dirname(__file__), "openai-key")
print(api_key_file)

# Read the API key from the file or prompt the user
if os.path.exists(api_key_file):
    with open(api_key_file, 'r') as file:
        api_key = file.read().strip()

client = OpenAI(api_key=api_key)

def ask_openai(question: str) -> str:
    response = client.completions.create(
      model="gpt-3.5-turbo-instruct",
      prompt=question,
      max_tokens=200
    )
    return response.choices[0].text.strip()

print(ask_openai("What is the capital of France?"))


from redisvl.extensions.llmcache import SemanticCache

llmcache = SemanticCache(
    name="llmcache",                     # underlying search index name
    redis_url="redis://localhost:6379",  # redis connection url string
    distance_threshold=0.1               # semantic cache distance threshold
)

question = "What is the capital of France?"
# Check the semantic cache -- should be empty
if response := llmcache.check(prompt=question):
    print(response)
else:
    print("Empty cache")

# Cache the question, answer, and arbitrary metadata
llmcache.store(
    prompt=question,
    response="Paris",
    metadata={"city": "Paris", "country": "france"}
)


# Check the cache again
if response := llmcache.check(prompt=question, return_fields=["prompt", "response", "metadata"]):
    print(response)
else:
    print("Empty cache")

# Check for a semantically similar result
question = "What actually is the capital of France?"
llmcache.check(prompt=question)[0]['response']