from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model_id = "gg-hf/gemma-2b-it"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=dtype,
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

