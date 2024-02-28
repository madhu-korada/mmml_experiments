import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
import torch
from torch.cuda import get_device_properties


def check_gpu_vram_less_than_8GB():
    if not torch.cuda.is_available():
        return False
    gpu_props = get_device_properties(0) # Assuming single GPU setup
    total_memory_GB = gpu_props.total_memory / (1024**3) # Convert bytes to GB
    print(f"Total GPU memory: {total_memory_GB:.2f} GB")
    return total_memory_GB < 8

if check_gpu_vram_less_than_8GB():
    print("GPU VRAM less than 8GB. Using 8-bit quantization.")
    os.environ['LD_LIBRARY_PATH'] = ''
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
else:
    quantization_config = None


model_id = "gg-hf/gemma-2b-it"
# model_id = "meta-llama/Llama-2-7b-chat-hf"
# model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# model_id = "mistralai/Mistral-7B-v0.1"
# model_id = "gpt2"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=dtype,
    use_auth_token=True,
    quantization_config=quantization_config
)

prompt = "Please answer the question according to the context and candidate answers. Each candidate answer is associated with a confidence score within a bracket. The true answer may not be included in the candidate answers. Answer the Question with a single word, if you don't know say NO. \n\n===\nContext: Two people on motorbikes very high in the air.\n===\nQuestion: What type of bikes are these people riding?\n===\nCandidates: motorbike(0.76), motorcycle(0.73), motocross(0.57), dirt bike(0.41), bmx(0.26), motorcross(0.21), dirt(0.15), motor(0.08), chopper(0.01), bicycle(0.01)\n===\nAnswer: motorbike"





chat = [
    { "role": "user", "content": {prompt} },
]

prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

# print(prompt)
# prompt = "Hello world"

inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=5)

print("\n\nOutput:")

print(tokenizer.decode(outputs[0]))

print(tokenizer.decode(outputs[-3:]))
