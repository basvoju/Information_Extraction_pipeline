from transformers import BertForTokenClassification, Trainer, TrainingArguments, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support

# Define the compute_metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# Load dataset
dataset = load_dataset('conll2003')

# Load the tokenizer and model
model_identifier = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_identifier)
model = BertForTokenClassification.from_pretrained(model_identifier, num_labels=9)  # Assuming 9 labels for token classification

# Tokenize the dataset with padding and truncation
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True,
        padding='max_length',
        max_length=128,  # Adjust max_length as needed
        is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        # Truncate or pad label_ids to match max_length
        label_ids = label_ids[:128] + [-100] * (128 - len(label_ids))
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Set training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',  # Use 'eval_strategy' instead of 'evaluation_strategy'
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()


from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load the fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-biobert")
model = AutoModelForTokenClassification.from_pretrained("./fine-tuned-biobert")

# Create a NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Example transcribed text
transcribed_text = r'C:/Users/BASVOJU/Desktop/Master_thesis/Transcriptions/transcribed_text.txt'

# Perform NER
ner_results = ner_pipeline(transcribed_text)

# Print the extracted medical entities
for entity in ner_results:
    print(f"Entity: {entity['word']}, Label: {entity['entity']}")
