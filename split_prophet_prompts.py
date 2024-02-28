import os
import json

json_folder = "prophet/outputs/results/okvqa_prompt_1/"
file_name = "cache.json"
json_path = json_folder + file_name

out = file_name.replace(".json", "")
output_folder = json_folder + f"split_{out}/"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

with open(json_path, "r") as f:
    prompts = json.load(f)
    prompts = list(prompts.values())
    
    num_parts = 5
    num_prompts = len(prompts)
    part_size = num_prompts // num_parts
    
    split_prompts = [prompts[i:i+part_size] for i in range(0, num_prompts, part_size)]

    for i, split_prompt in enumerate(split_prompts):
        new_json_path = output_folder + file_name.replace(".json", f"_split_{i+1}.json")
        with open(new_json_path, "w") as f:
            json.dump(split_prompt, f)


