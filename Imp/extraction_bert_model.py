import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load the fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

# Create a NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Example transcribed text
transcribed_text = r'C:/Users/BASVOJU/Desktop/Master_thesis/Transcriptions/transcribed_text.txt'

# Perform NER
ner_results = ner_pipeline(transcribed_text)

# Print the extracted medical entities
for entity in ner_results:
    print(f"Entity: {entity['word']}, Label: {entity['entity']}")
