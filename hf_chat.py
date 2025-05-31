from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-small"

# Load model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def hf_chat(messages):
    # Use only the latest message
    user_msg = messages[-1].content

    # Prepare the prompt
    prompt = f"Answer the following clearly:\n{user_msg}"

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=100)

    # Decode and return
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()
