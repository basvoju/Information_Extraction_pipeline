from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# Load the BERT model and tokenizer
model_name = "nlpaueb/bert-base-german-uncased"  # Replace with the correct model name if different
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Initialize the NER pipeline with the BERT model
ner_model = pipeline("ner", model=model, tokenizer=tokenizer)

def extract_medical_entities(text):
    """
    Extract medical entities from text using NER model.
    """
    ner_results = ner_model(text)
    medical_entities = [result['word'] for result in ner_results if 'medical' in result['entity'].lower()]
    return medical_entities

def extract_from_transcription(transcribed_file_path):
    """
    Extract medical entities from the transcribed text file.
    """
    with open(transcribed_file_path, "r", encoding="iso-8859-1") as f:
        transcribed_text = f.read().replace("\n", " ")

    medical_entities = extract_medical_entities(transcribed_text)
    print("Extracted Medical Entities:", medical_entities)

if __name__ == "__main__":
    # Path to the transcribed text file
    transcribed_file_path = r'C:/Users/BASVOJU/Desktop/Master_thesis/Transcriptions/transcribed_text.txt'

    # Extract medical entities from the transcribed text
    extract_from_transcription(transcribed_file_path)
