import os
import requests
from flask import Flask, render_template, request, jsonify, send_from_directory
from io import BytesIO
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import psutil
from werkzeug.utils import secure_filename
from datasets import load_dataset

# --- CONFIG ---
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
AI_MODE = 'local'  # 'local' or 'api'
HF_API_MODEL = "microsoft/DialoGPT-small"
HF_TOKEN_FILE = os.path.join(os.path.dirname(__file__), 'templates', 'huggingface.txt')

# --- APP INIT ---
app = Flask(__name__)

# --- LOCAL PIPELINE SETUP ---
if AI_MODE == 'local':
    model_id = "microsoft/DialoGPT-small"
    pipe = pipeline("text-generation", model=model_id)

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/templates/style.css')
def style():
    return send_from_directory(os.path.join(app.root_path, 'templates'), 'style.css')

@app.route('/api/generate', methods=['POST'])
def generate_text():
    data = request.get_json(force=True)
    prompt = data.get('prompt', 'Explain quantum mechanics clearly and concisely.')
    try:
        if AI_MODE == 'local':
            outputs = pipe(prompt, max_new_tokens=256)
            ai_response = outputs[0]["generated_text"]
        else:
            return jsonify({'error': 'Not supported in API mode.'}), 400
    except Exception as e:
        ai_response = f"Error: {str(e)}"
    return jsonify({'response': ai_response})

@app.route('/api/image-text', methods=['POST'])
def image_text_to_text():
    if 'image' not in request.files or 'prompt' not in request.form:
        return jsonify({'error': 'Image file and prompt are required.'}), 400

    image_file = request.files['image']
    prompt = request.form['prompt']

    try:
        if os.path.exists(HF_TOKEN_FILE):
            with open(HF_TOKEN_FILE) as f:
                hf_token = f.read().strip()
        else:
            hf_token = os.environ.get('HF_TOKEN', '')

        headers = {"Authorization": f"Bearer {hf_token}"}
        files = {
            'image': (secure_filename(image_file.filename), image_file.read(), image_file.mimetype),
        }
        data = {'inputs': prompt}
        api_url = "https://api-inference.huggingface.co/models/google/gemma-3n-e4b-it"

        resp = requests.post(api_url, headers=headers, data=data, files=files, timeout=60)
        resp.raise_for_status()
        result = resp.json()

        if isinstance(result, dict) and 'generated_text' in result:
            ai_response = result['generated_text']
        elif isinstance(result, list) and 'generated_text' in result[0]:
            ai_response = result[0]['generated_text']
        elif isinstance(result, dict) and 'error' in result:
            ai_response = f"API Error: {result['error']}"
        else:
            ai_response = str(result)

    except Exception as e:
        ai_response = f"Error: {str(e)}"

    return jsonify({'response': ai_response})

@app.route('/api/ask', methods=['POST'])
def ask():
    user_input = request.json.get('question')

    try:
        if AI_MODE == 'local':
            outputs = pipe(user_input, max_new_tokens=256)
            ai_response = outputs[0]['generated_text']
        else:
            system_prompt = "You are a helpful, friendly AI assistant."
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]

            if os.path.exists(HF_TOKEN_FILE):
                with open(HF_TOKEN_FILE) as f:
                    hf_token = f.read().strip()
            else:
                hf_token = os.environ.get('HF_TOKEN', '')

            headers = {"Authorization": f"Bearer {hf_token}"}
            payload = {"inputs": messages}
            api_url = f"https://api-inference.huggingface.co/models/{HF_API_MODEL}"

            resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, list) and 'generated_text' in data[0]:
                ai_response = data[0]['generated_text']
            elif isinstance(data, dict) and 'generated_text' in data:
                ai_response = data['generated_text']
            elif isinstance(data, dict) and 'error' in data:
                ai_response = f"API Error: {data['error']}"
            else:
                ai_response = str(data)
    except Exception as e:
        ai_response = f"Error: {str(e)}"

    return jsonify({'response': ai_response})

@app.route('/api/system-stats', methods=['GET'])
def system_stats():
    stats = {
        'cpu_percent': psutil.cpu_percent(interval=0.5),
        'memory': psutil.virtual_memory()._asdict(),
        'disk': psutil.disk_usage('/')._asdict(),
        'loadavg': os.getloadavg() if hasattr(os, 'getloadavg') else None
    }
    return jsonify(stats)

# Load pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Set pad_token if not already set
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Load your dataset (assumes a text file with one line per training example)
data_files = {"train": "/workspaces/Education-Application-With-AI-Integration/app/modules/dataset.txt"}  # Make sure this file exists
raw_datasets = load_dataset("text", data_files=data_files)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["text"])

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # No masked language modeling; we want causal LM like GPT
)


# --- MAIN ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5600, debug=True)
