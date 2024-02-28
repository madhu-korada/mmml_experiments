import os
import json
import time
import torch
import transformers
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.cuda import get_device_properties
from torch.nn.functional import softmax

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
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False, return_tensors="pt")
        output_sequences = self.model.generate(input_ids=input_ids.to(self.model.device), max_new_tokens=5, output_scores=True)
        return output_sequences
    
    def get_answer_given_prompt(self, prompt, num_return_sequences=5):
        """
        Generates text based on the input text.
        
        Args:
        input_text (str): The input text to complete.
        max_length (int): The maximum length of the sequence to be generated.
        num_return_sequences (int): The number of sequences to generate.
        
        Returns:
        List[str]: The generated text sequences.
        """
        output_sequences = self.generate_text(prompt, num_return_sequences=num_return_sequences)
        # Decode and return the generated sequences
        output = hf_model.tokenizer.decode(output_sequences[0])
        answer = output.split("Answer: ")[-1].strip().replace("<eos>", "")
            
        return answer
    


model_id = "gg-hf/gemma-2b-it"
# model_id = "meta-llama/Llama-2-7b-chat-hf"
# model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# model_id = "mistralai/Mistral-7B-v0.1"
# model_id = "gpt2"
dtype = torch.bfloat16
hf_model = HFModel(model_id)


files_to_process = [1]
for file_to_process in files_to_process:
    prompts_json_file = f"prophet/outputs/results/okvqa_prompt_1/split_cache/cache_split_{file_to_process}.json"
    inference_batch_size = 5

    prompts_with_answers_json_file = prompts_json_file.replace(".json", "_with_answers.json")

    # load the prompts from the json file
    with open(prompts_json_file, "r") as f:
        prompts = json.load(f)
        print("Prompts in this file:", len(prompts))
        new_items = []
        i = 0
        for prompt in prompts:
            i+=1
            print(f"Processing prompt {i+1}/{len(prompts)}")
            question_id = prompt['question_id']
            prompt_for_model = prompt['prompt_info'][0]['prompt']
            
            start_time = time.time()
            answer = hf_model.get_answer_given_prompt(prompt_for_model, num_return_sequences=5)

            print(answer)
            print("Time taken:", time.time() - start_time)

            # construct the prompt with the answer
            new_prompt = {
                "question_id": question_id,
                "answer": answer,
                "prompt_info": prompt['prompt_info']
            }
            new_items.append(new_prompt)
        
        # save the prompts with the answers
        with open(prompts_with_answers_json_file, "w") as f:
            json.dump(new_items, f)
        print("Prompts with answers saved to:", prompts_with_answers_json_file)
        