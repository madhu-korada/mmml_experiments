

# write a function to get context, question, candidates, and answer from the files




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


class HFModel:
    def __init__(self, model_name: str):
        """
        Initializes the LLaMA model and tokenizer.
        
        Args:
        model_name (str): The model identifier from Hugging Face's model hub.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            use_auth_token=True,
            quantization_config=quantization_config
        )

    def generate_text(self, input_text: str, max_length: int = 50, num_return_sequences: int = 1):
        """
        Generates text based on the input text.
        
        Args:
        input_text (str): The input text to complete.
        max_length (int): The maximum length of the sequence to be generated.
        num_return_sequences (int): The number of sequences to generate.
        
        Returns:
        List[str]: The generated text sequences.
        """
        # input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        # output_sequences = self.model.generate(input_ids, max_length=max_length, num_return_sequences=num_return_sequences)
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False, return_tensors="pt")
        output_sequences = self.model.generate(input_ids=input_ids.to(self.model.device), max_new_tokens=5)
        # # Decode and return the generated sequences
        # return [self.tokenizer.decode(output_seq, skip_special_tokens=True) for output_seq in output_sequences]
        return output_sequences




model_id = "gg-hf/gemma-2b-it"
# model_id = "meta-llama/Llama-2-7b-chat-hf"
# model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# model_id = "mistralai/Mistral-7B-v0.1"
# model_id = "gpt2"
dtype = torch.bfloat16



# Example usage:]
hf_model = HFModel(model_id)

input_text = "The future of AI is"
generated_texts = hf_model.generate_text(input_text, max_length=100, num_return_sequences=3)

for text in generated_texts:
    print(text)


prompt = "Please answer the question according to the context and candidate answers. Each candidate answer is associated with a confidence score within a bracket. The true answer may not be included in the candidate answers. Answer the Question no matter what. Reply in the format: Answer: <answer>"
#  Reply in the format: Answer: <answer>

context = "\n\n===\nContext: Two people on motorbikes very high in the air."
question = "\n===\nQuestion: What type of bikes are these people riding?"
candidates = "\n===\nCandidates: motorbike(0.76), motorcycle(0.73), motocross(0.57), dirt bike(0.41), bmx(0.26), motorcross(0.21), dirt(0.15), motor(0.08), chopper(0.01), bicycle(0.01)\n===\n"
answer = "Answer: motorbike."

complete = prompt + context + question + candidates + answer
chat = [
    { "role": "user", "content": {complete} },
]
print(complete)
# prompt = hf_model.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
prompt = complete

output_sequences = hf_model.generate_text(prompt, num_return_sequences=5)
print(output_sequences)

print("\n\nOutput:")
# output = hf_model.tokenizer.decode(outputs[0])
output = hf_model.tokenizer.decode(output_sequences[0])
print(output)
print("\n\n")
split_output = output.split("Answer: ")
print(split_output[-1])



