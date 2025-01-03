import os
import torch
from transformers import pipeline
import time

# Load the local LLaMA model
model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Function to interact with the LLaMA model
def ask_local_llama(question: str) -> str:
    # Create input in the format expected by the pipeline
    input_text = f"System: You are an assistant. Answer the following question concisely.\nUser: {question}\nAssistant:"
    outputs = pipe(
        input_text,
        max_new_tokens=256,
    )
    return outputs[0]["generated_text"].strip()

# Example query to the local LLaMA model
question = "What is the capital of China?"
start = time.time()
print("LLaMA Response:", ask_local_llama(question))
print ("llm time:" + str(time.time() - start))

# Integrating the response with the Redis-based semantic cache
from redisvl.extensions.llmcache import SemanticCache

# Initialize Semantic Cache
llmcache = SemanticCache(
    name="llmcache",                     # underlying search index name
    redis_url="redis://localhost:6379",  # redis connection URL
    distance_threshold=0.1               # semantic cache distance threshold
)

# Query the cache or fetch from the LLaMA model
response = llmcache.check(prompt=question)
if response:
    print("Cache Hit:", response[0]["response"])
else:
    print("Cache Miss. Querying LLaMA...")
    response_text = ask_local_llama(question)
    print("Response:", response_text)
    llmcache.store(
        prompt=question,
        response=response_text,
        metadata={"source": "local_llama"}
    )

# Demonstrate semantic similarity check
similar_question = "What actually is the capital of France?"
similar_response = llmcache.check(prompt=similar_question)
if similar_response:
    print("Semantically Similar Cache Hit:", similar_response[0]["response"])
else:
    print("No similar response in cache.")
