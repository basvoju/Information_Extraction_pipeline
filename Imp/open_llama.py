from transformers import LlamaTokenizer, LlamaForCausalLM

# Load the LLaMA model and tokenizer
llama_model_name = "meta-llama/Llama-7b"  # Replace with the exact model name if different
tokenizer = LlamaTokenizer.from_pretrained(llama_model_name)
model = LlamaForCausalLM.from_pretrained(llama_model_name)

def extract_medical_info(text):
    # Define a prompt to guide the model in extracting relevant information
    prompt = f"Extract important medical information from the following german text, this text is the conversation in a ambulance between the patient and the first responder in the ambulance, now this data which is generated from the speech to text model is given here and need to extract useful medical information from the text so that doctor can treat the patient effiently:\n\n{text}\n\nImportant medical information:"

    # Tokenize and generate the model's response
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=500, num_return_sequences=1)
    extracted_info = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return extracted_info


def main():
    # Example transcribed text (replace with your actual transcribed text)
    transcribed_text = r'C:/Users/BASVOJU/Desktop/Master_thesis/Transcriptions/transcribed_text.txt'


    # Extract important medical information
    extracted_info = extract_medical_info(transcribed_text)
    print(f"Extracted Medical Information: {extracted_info}")


if __name__ == "__main__":
    main()
