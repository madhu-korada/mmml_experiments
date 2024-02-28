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
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
else:
    quantization_config = None


model_id = "gg-hf/gemma-2b-it"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=dtype,
    use_auth_token=True,
    quantization_config=quantization_config
)

chat = [
    { "role": "user", "content": "Write a hello world program" },
]

prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

# print(prompt)
# prompt = "Hello world"

inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=150)
print(tokenizer.decode(outputs[0]))

