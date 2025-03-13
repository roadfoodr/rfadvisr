import json
import tiktoken
import numpy as np
from collections import defaultdict
import os

# Configuration constants
TRAIN_JSONL = 'data/roadfood_finetune_train.jsonl'
TEST_JSONL = 'data/roadfood_finetune_test.jsonl'

def load_dataset(data_path):
    """Load the dataset from a JSONL file."""
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]
        
        print(f"Loaded {len(dataset)} examples from {data_path}")
        print("First example:")
        for message in dataset[0]["messages"]:
            print(message)
        
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

def validate_format(dataset):
    """Validate the format of the dataset."""
    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue
            
        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue
            
        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1
            
            if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
                format_errors["message_unrecognized_key"] += 1
            
            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1
                
            content = message.get("content", None)
            function_call = message.get("function_call", None)
            
            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1
        
        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("\nFound format errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("\nNo format errors found")
    
    return format_errors

def num_tokens_from_messages(messages, encoding, tokens_per_message=3, tokens_per_name=1):
    """Count the number of tokens in a list of messages."""
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens

def num_assistant_tokens_from_messages(messages, encoding):
    """Count the number of tokens in assistant messages."""
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens

def print_distribution(values, name):
    """Print distribution statistics for a list of values."""
    print(f"\n#### Distribution of {name}:")
    print(f"min / max: {min(values)}, {max(values)}")
    print(f"mean / median: {np.mean(values):.2f}, {np.median(values):.2f}")
    print(f"p10 / p90: {np.quantile(values, 0.1):.2f}, {np.quantile(values, 0.9):.2f}")

def analyze_dataset(dataset):
    """Analyze the dataset for warnings and token counts."""
    encoding = tiktoken.get_encoding("cl100k_base")
    
    n_missing_system = 0
    n_missing_user = 0
    n_messages = []
    convo_lens = []
    assistant_message_lens = []

    for ex in dataset:
        messages = ex["messages"]
        if not any(message["role"] == "system" for message in messages):
            n_missing_system += 1
        if not any(message["role"] == "user" for message in messages):
            n_missing_user += 1
        n_messages.append(len(messages))
        convo_lens.append(num_tokens_from_messages(messages, encoding))
        assistant_message_lens.append(num_assistant_tokens_from_messages(messages, encoding))
        
    print("\n### Dataset Analysis:")
    print(f"Num examples missing system message: {n_missing_system}")
    print(f"Num examples missing user message: {n_missing_user}")
    
    print_distribution(n_messages, "num_messages_per_example")
    print_distribution(convo_lens, "num_total_tokens_per_example")
    print_distribution(assistant_message_lens, "num_assistant_tokens_per_example")
    
    n_too_long = sum(l > 16385 for l in convo_lens)
    print(f"\n{n_too_long} examples may be over the 16,385 token limit, they will be truncated during fine-tuning")
    
    return convo_lens

def estimate_cost(convo_lens):
    """Estimate the cost of fine-tuning based on token counts."""
    MAX_TOKENS_PER_EXAMPLE = 16385

    TARGET_EPOCHS = 3
    MIN_TARGET_EXAMPLES = 100
    MAX_TARGET_EXAMPLES = 25000
    MIN_DEFAULT_EPOCHS = 1
    MAX_DEFAULT_EPOCHS = 25

    n_epochs = TARGET_EPOCHS
    n_train_examples = len(convo_lens)
    
    if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
        n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
    elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
        n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

    n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)
    
    print("\n### Cost Estimation:")
    print(f"Dataset has ~{n_billing_tokens_in_dataset:,} tokens that will be charged for during training")
    print(f"By default, you'll train for {n_epochs} epochs on this dataset")
    print(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset:,} tokens")
    print("\nSee https://openai.com/pricing to estimate total costs.")

def main():
    # Validate training data
    print("=" * 80)
    print(f"Validating training data: {TRAIN_JSONL}")
    print("=" * 80)
    train_dataset = load_dataset(TRAIN_JSONL)
    if train_dataset:
        train_errors = validate_format(train_dataset)
        if not train_errors:
            train_convo_lens = analyze_dataset(train_dataset)
            estimate_cost(train_convo_lens)
    
    # Validate test data
    print("\n" + "=" * 80)
    print(f"Validating test data: {TEST_JSONL}")
    print("=" * 80)
    test_dataset = load_dataset(TEST_JSONL)
    if test_dataset:
        test_errors = validate_format(test_dataset)
        if not test_errors:
            analyze_dataset(test_dataset)

if __name__ == "__main__":
    main() 