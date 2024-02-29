# mmml_experiments
## For Prophet

```
bash scripts/prompt.sh --task ok --version okvqa_prompt_1 --openai_key sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```


```
python -m llava.eval.model_vqa  --model-path liuhaotian/llava-v1.5-7b  --question-file ok_vqa_questions.jsonl  --image-folder  data/coco2014/val2014/ --answers-file llava/eval/answer-file-our-single-word-final.jsonl
```

```
python -m llava.eval.model_vqa  --model-path liuhaotian/llava-v1.5-7b  --question-file ../questions.txt  --image-folder  data/coco2014/val2014/ --answers-file llava/eval/answer-file-our-single-word-temp-1-beams-5.jsonl
```

