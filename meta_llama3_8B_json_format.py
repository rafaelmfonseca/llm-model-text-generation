import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from jsonformer import Jsonformer
from huggingface_hub import login
from dotenv import load_dotenv
import torch

load_dotenv(".env.local")

huggingface_api_key = os.environ.get("HUGGINGFACE_API_KEY")
login(huggingface_api_key)

print("Is CUDA enabled? ", torch.cuda.is_available())
print("Torch current device: ", torch.cuda.current_device())

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"},
        "is_student": {"type": "boolean"},
        "courses": {
            "type": "array",
            "items": {"type": "string"}
        }
    }
}

prompt = "Generate a person's information based on the following schema:"
jsonformer = Jsonformer(model, tokenizer, json_schema, prompt)
generated_data = jsonformer()

print(generated_data)