from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


intents = {
    "troubleshooting": ["fix", "problem", "not working", "error", "issue"],
    "account_info": ["account", "login", "password", "username", "authentication"],
    "service_status": ["service down", "status", "outage", "available"],
    "upgrade": ["upgrade", "update", "new version", "latest version"]
}

def generate_backup_response(prompt):
    response = ''
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=100
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            response += chunk.choices[0].delta.content
    return response

def identify_intent(query):
    for intent, keywords in intents.items():
        if any(keyword in query.lower() for keyword in keywords):
            return intent
    return "general"  # default intent if no specific intent is found

def clean_text(text):
    """ Basic cleaning of texts. """
    return text.strip().lower()

def generate_response(text, intent, max_attempts=10):
    """Generates a response ensuring it does not contain the original input text."""
    attempts = 0
    input_text = f"{tokenizer.eos_token}{text}"
    while attempts < max_attempts:
        prompt = f"{intent.capitalize()} query: {input_text}"
        inputs = tokenizer.encode_plus(
            input_text,
            return_tensors='pt',
            max_length=30,  # Limit input length
            padding='max_length',  # Pad to max_length
            truncation=True,
            add_special_tokens=True
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        reply_ids = model.module.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100,  # Generate up to 100 tokens
            pad_token_id=tokenizer.eos_token_id,
            top_p=0.92,
            top_k=50,
            temperature=0.8,
            do_sample=True,  # Enable sampling for more diverse responses
            num_return_sequences=1
        )

        reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

        # Check if the reply contains the input text
        if clean_text(text) not in clean_text(reply):
            return reply
        attempts += 1
    if attempts >= max_attempts:
        reply = generate_backup_response(text)
    return reply  # Return the last attempt even if not perfect


# Load the model and tokenizer
model = GPT2LMHeadModel.from_pretrained('./static/model_final')
# Check if CUDA is available and move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model = torch.nn.DataParallel(model)
tokenizer = GPT2Tokenizer.from_pretrained('./static/model_final')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set for padding
tokenizer.padding_side = 'left'  # Ensure padding is done on the left
model.eval()  # Set model to evaluation mode to deactivate dropout layers

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/send', methods=['POST'])
def send():
    text = request.json['message']
    intent = identify_intent(text)
    response = generate_response(text, intent)
    return jsonify({'reply': response})

if __name__ == '__main__':
    app.run(debug=True)
