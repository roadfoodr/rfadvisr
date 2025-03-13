# OpenAI Fine-tuning for Roadfood Dataset

This directory contains scripts for creating and testing OpenAI fine-tuned models based on the Roadfood dataset.

## Files

- `create_openai_finetune.py`: Creates JSONL files for fine-tuning with train/test split and optionally submits a fine-tuning job to OpenAI
- `test_finetune_model.py`: Tests a fine-tuned model with samples from the dataset or custom prompts

## Setup

1. Make sure you have the required dependencies installed:
   ```
   pip install pandas openai pyyaml scikit-learn
   ```

2. Ensure you have a `credentials.yml` file in the root directory with your OpenAI API key:
   ```yaml
   openai: "sk-your-api-key"
   ```

## Creating a Fine-tuning Dataset

The `create_openai_finetune.py` script is configured to:

1. Load the Roadfood dataset
2. Filter out rows where `Crossout` is 'y'
3. Randomly select 250 rows from the remaining data
4. Split the data into training (80%) and test (20%) sets
5. Create separate JSONL files for training and testing in the format required by OpenAI

By default, the script is set to only create the JSONL files without submitting a fine-tuning job.

### Configuration

You can modify the following constants at the top of the script:

```python
INPUT_CSV = 'data/Roadfood_10th_supplemented.csv'    # Input dataset
OUTPUT_TRAIN_JSONL = 'data/roadfood_finetune_train.jsonl'  # Training data output
OUTPUT_TEST_JSONL = 'data/roadfood_finetune_test.jsonl'    # Test data output
PROMPT_COL = 'summary_q'                             # Column to use as prompt
COMPLETION_COL = 'content'                           # Column to use as completion
BASE_MODEL = 'gpt-3.5-turbo'                         # Base model to fine-tune
SAMPLE_SIZE = 250                                    # Number of examples to include
TEST_SIZE = 0.2                                      # Proportion of data for testing
CREATE_ONLY = True                                   # Just create JSONL, don't submit job
```

### Running the Script

```
python wrangle/create_openai_finetune.py
```

## Testing a Fine-tuned Model

Once you have a fine-tuned model, you can test it using the `test_finetune_model.py` script.

### Configuration

Update the following constants at the top of the script:

```python
MODEL_ID = "ft:gpt-3.5-turbo-0613:personal::abc123"  # Your fine-tuned model ID
INPUT_CSV = 'data/Roadfood_10th_supplemented.csv'    # Dataset for test samples
PROMPT_COL = 'summary_q'                             # Column to use as prompt
NUM_SAMPLES = 5                                      # Number of samples to test
TEMPERATURE = 0.7                                    # Temperature for generation
```

### Running the Script

```
python wrangle/test_finetune_model.py
```

## Fine-tuning Process

1. Create the JSONL files using `create_openai_finetune.py`
2. Submit the fine-tuning job through the OpenAI API (set `CREATE_ONLY = False`)
3. Wait for the fine-tuning job to complete (typically takes 1-4 hours)
4. Test the fine-tuned model using `test_finetune_model.py`

## Notes

- The fine-tuning is set up for single-turn completions, not multi-turn chats
- The scripts use the `summary_q` column as the prompt and `content` column as the completion
- The training/test split follows OpenAI's best practices for evaluating model performance
- You can modify the scripts to use different columns or formats as needed 