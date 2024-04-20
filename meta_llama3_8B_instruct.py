import os

from transformers import AutoTokenizer, AutoModelForCausalLM
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
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a developer that write codes for me. I ask you to write a code in csharp and you send to me just the code."},
    {"role": "user", "content": "Write a code that recusively find zipped files in a directory and subdirectories."},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
output = tokenizer.decode(response, skip_special_tokens=True)

print(output)