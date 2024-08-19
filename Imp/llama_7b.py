from transformers import AutoTokenizer, AutoModelForCausalLM

# Replace with a valid model identifier
model_name = 'meta-llama/Llama-7b'  # Example model identifier

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Example text input
input_text = r'C:/Users/BASVOJU/Desktop/Master_thesis/Transcriptions/transcribed_text.txt'

# Tokenize and generate
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)

# Decode and print the result
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded_output)
