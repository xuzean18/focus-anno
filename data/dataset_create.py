from datasets import load_dataset
import pandas as pd
import os
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

# Download NLTK data
nltk.download('punkt')

# Load the SQuAD dataset from Huggingface
squad = load_dataset('squad')

# Create a function to process each entry in SQuAD
def process_qc1a_entry(entry, index):
    global context_found
    question = entry['question']
    context_sentences = entry['context']
    answer = entry['answers']['text'][0]

    # Find the one period-marked context sentence that includes the entire answer phrase
    for context_sent in context_sentences.split('.'):
        if context_sent.find(answer) != -1:
            context_found = context_sent 
            break
    
    # Locate the position of the answer within the context sentence
    answer_start = context_found.find(answer)
    answer_end = answer_start + len(answer)
    
    # Tokenize the context sentence
    context_tokens = word_tokenize(context_found)
    
    # Initialize the labels list with all 'O'
    ca_labels = ['O'] * len(context_tokens)
    
    # Mark tokens in the answer with 'B-focus' and 'I-focus'
    current_position = 0
    token_positions = []
    for token in context_tokens:
        token_start = context_found.lower().find(token.lower(), current_position)
        token_end = token_start + len(token)
        token_positions.append((token_start, token_end))
        current_position = token_end
    
    answer_found = False
    for i, (token_start, token_end) in enumerate(token_positions):
        if token_start >= answer_start and token_end <= answer_end:
            if not answer_found:
                ca_labels[i] = 'B-focus'
                answer_found = True
            else:
                ca_labels[i] = 'I-focus'
                
    # Tokenize the question
    question_tokens = word_tokenize(question)
    question_context_tokens = question_tokens + context_tokens
    
    # Mark the tokens of the question with all 'O'
    qa_labels = ['O'] * len(question_tokens) + ca_labels
    
    return {'id': str(index), 'tokens': question_context_tokens, 'ner_tags': qa_labels}

# Create a list of the training data
train_data = [squad['train'][index] for index in range(len(squad['train']))]

# Create a list of the validation data
val_data = [squad['validation'][index] for index in range(len(squad['validation']))]

# Process the training data for QA combination
processed_qc1a_train = [process_qc1a_entry(entry, index) for index, entry in enumerate(train_data)]

# Process the validation data for QA combination
processed_qc1a_val = [process_qc1a_entry(entry, index) for index, entry in enumerate(val_data)]

# Create dataframes for training and validation sets
train_data = pd.DataFrame(processed_qc1a_train)
val_data = pd.DataFrame(processed_qc1a_val)

# Split half of the validation set for test
val_data, test_data = train_test_split(val_data, test_size=0.5, shuffle=True, random_state=42)

# Save the dataframes as TSV files on the desktop
output_dir = os.path.expanduser("~/Desktop")
train_data.to_csv(os.path.join(output_dir, 'train.tsv'), sep='\t', index=False)
val_data.to_csv(os.path.join(output_dir, 'val.tsv'), sep='\t', index=False)
test_data.to_csv(os.path.join(output_dir, 'test.tsv'), sep='\t', index=False)