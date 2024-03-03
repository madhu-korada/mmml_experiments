from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers

def calculate_f1_score(predicted_token, true_token):
    """
    Calculate the F1 score for a question answering task.
    
    Args:
    predicted_answer (str): The predicted answer.
    true_answer (str): The true answer.
    
    Returns:
    float: The F1 score.
    """
    # import pdb
    # pdb.set_trace()
    # Tokenize the predicted and true answers
    predicted_tokens = set(predicted_token)
    true_tokens = set(true_token)
    # print(f'Predicted: {predicted_tokens} True: {true_tokens}')
    # Calculate common tokens
    common_tokens = predicted_tokens & true_tokens
    
    # Calculate precision
    precision = len(common_tokens) / len(predicted_tokens) if len(predicted_tokens) > 0 else 0
    
    # Calculate recall
    recall = len(common_tokens) / len(true_tokens) if len(true_tokens) > 0 else 0
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1_score, precision, recall
    
def f1(prediction, ground_truths, tokenizer):
    # print(f'Prediction: {prediction} Ground Truth: {ground_truths}')
    f1_score = 0
    precision = 0
    recall = 0
    for i, gt in enumerate(ground_truths):
        # print(f'Prediction: {prediction} Ground Truth: {gt}')
        # import pdb
        # pdb.set_trace()
        tmp_f1_score, tmp_precision, tmp_recall = calculate_f1_score(tokenizer.encode(prediction), tokenizer.encode(gt))
        # tmp_f1_score, tmp_precision, tmp_recall = calculate_f1_score(prediction, gt)
        f1_score += tmp_f1_score
        precision += tmp_precision
        recall += tmp_recall
    cur_f1 = min(float(f1_score/len(ground_truths)), 1.0)
    cur_precision = min(float(precision/len(ground_truths)), 1.0)
    cur_recall = min(float(recall/len(ground_truths)), 1.0)
    return cur_f1, cur_precision, cur_recall



if __name__ == "__main__":
        
    gold =  ['africa country', 'africa', 'africa', 'africa', 'africa', 'africa', 'africa', 'africa', 'zoo', 'zoo']
    model_id = "gg-hf/gemma-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id, is_split_into_words=True)
    # print(f1(['africa conutry', 'africa'], gold, tokenizer))
    
    pred = ['wetsuit', 'suit']#, 'wet suit']
    gold = ['wetsuit', 'wetsuit', 'wetsuit', 'wetsuit', 'suit', 'suit', 'wet suit', 'wet suit', 'trunk', 'trunk']
    print(f1(pred, gold, tokenizer))

