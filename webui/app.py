import os
import uuid
import json
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename
from kittentts import KittenTTS

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, 'static', 'audio')
HISTORY_FILE = os.path.join(BASE_DIR, 'history.json')

os.makedirs(AUDIO_DIR, exist_ok=True)

AVAILABLE_MODELS = [
    {"id": "KittenML/kitten-tts-mini-0.8", "name": "Mini (80M)", "description": "最高质量版本"},
    {"id": "KittenML/kitten-tts-micro-0.8", "name": "Micro (40M)", "description": "速度与质量平衡"},
    {"id": "KittenML/kitten-tts-nano-0.8", "name": "Nano (15M)", "description": "最小最快版本"},
]

MODELS = {}
MODELS_LOCK = threading.Lock()
TASK_QUEUE = []
TASK_QUEUE_LOCK = threading.Lock()
HISTORY = []
HISTORY_LOCK = threading.Lock()

def load_model(model_id):
    with MODELS_LOCK:
        if model_id in MODELS:
            return MODELS[model_id]
        print(f"Loading model: {model_id}")
        model = KittenTTS(model_id)
        MODELS[model_id] = model
        return model

def load_history():
    global HISTORY
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                HISTORY = json.load(f)
        except:
            HISTORY = []
    else:
        HISTORY = []

def save_history():
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(HISTORY, f, ensure_ascii=False, indent=2)

def add_to_history(task):
    global HISTORY
    with HISTORY_LOCK:
        HISTORY.insert(0, task)
        if len(HISTORY) > 100:
            HISTORY = HISTORY[:100]
        save_history()

def process_task(task):
    try:
        task['status'] = 'processing'
        model = load_model(task['model_id'])
        audio = model.generate(
            text=task['text'],
            voice=task['voice'],
            speed=task['speed'],
            clean_text=task.get('clean_text', True)
        )
        
        audio_filename = f"{task['id']}.wav"
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        import soundfile as sf
        sf.write(audio_path, audio, 24000)
        
        task['status'] = 'completed'
        task['audio_file'] = audio_filename
        task['completed_at'] = datetime.now().isoformat()
        
        with TASK_QUEUE_LOCK:
            for i, t in enumerate(TASK_QUEUE):
                if t['id'] == task['id']:
                    TASK_QUEUE[i] = task
                    break
        
        add_to_history(task)
    except Exception as e:
        task['status'] = 'failed'
        task['error'] = str(e)
        with TASK_QUEUE_LOCK:
            for i, t in enumerate(TASK_QUEUE):
                if t['id'] == task['id']:
                    TASK_QUEUE[i] = task
                    break

def task_worker():
    while True:
        task_to_process = None
        with TASK_QUEUE_LOCK:
            for task in TASK_QUEUE:
                if task['status'] == 'pending':
                    task_to_process = task
                    break
        
        if task_to_process:
            process_task(task_to_process)
        else:
            import time
            time.sleep(1)

worker_thread = threading.Thread(target=task_worker, daemon=True)
worker_thread.start()

load_history()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify(AVAILABLE_MODELS)

@app.route('/api/voices', methods=['GET'])
def get_voices():
    model_id = request.args.get('model', 'KittenML/kitten-tts-mini-0.8')
    try:
        model = load_model(model_id)
        voices = [
            {'id': 'Bella', 'name': 'Bella', 'description': '女声'},
            {'id': 'Jasper', 'name': 'Jasper', 'description': '男声'},
            {'id': 'Luna', 'name': 'Luna', 'description': '女声'},
            {'id': 'Bruno', 'name': 'Bruno', 'description': '男声'},
            {'id': 'Rosie', 'name': 'Rosie', 'description': '女声'},
            {'id': 'Hugo', 'name': 'Hugo', 'description': '男声'},
            {'id': 'Kiki', 'name': 'Kiki', 'description': '女声'},
            {'id': 'Leo', 'name': 'Leo', 'description': '男声'},
        ]
        return jsonify(voices)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        text = data.get('text', '')
        model_id = data.get('model', 'KittenML/kitten-tts-mini-0.8')
        voice = data.get('voice', 'Bruno')
        speed = float(data.get('speed', 1.0))
        clean_text = data.get('clean_text', True)
        
        if not text:
            return jsonify({'error': '文本不能为空'}), 400
        
        model = load_model(model_id)
        audio = model.generate(
            text=text,
            voice=voice,
            speed=speed,
            clean_text=clean_text
        )
        
        task_id = str(uuid.uuid4())
        audio_filename = f"{task_id}.wav"
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        import soundfile as sf
        sf.write(audio_path, audio, 24000)
        
        task = {
            'id': task_id,
            'text': text[:100] + ('...' if len(text) > 100 else ''),
            'model_id': model_id,
            'voice': voice,
            'speed': speed,
            'status': 'completed',
            'audio_file': audio_filename,
            'created_at': datetime.now().isoformat(),
            'completed_at': datetime.now().isoformat(),
        }
        
        add_to_history(task)
        
        return jsonify({
            'success': True,
            'task': task,
            'audio_url': f'/api/audio/{audio_filename}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tasks', methods=['POST'])
def add_task():
    try:
        data = request.get_json()
        text = data.get('text', '')
        model_id = data.get('model', 'KittenML/kitten-tts-mini-0.8')
        voice = data.get('voice', 'Bruno')
        speed = float(data.get('speed', 1.0))
        clean_text = data.get('clean_text', True)
        
        if not text:
            return jsonify({'error': '文本不能为空'}), 400
        
        task_id = str(uuid.uuid4())
        task = {
            'id': task_id,
            'text': text,
            'model_id': model_id,
            'voice': voice,
            'speed': speed,
            'clean_text': clean_text,
            'status': 'pending',
            'created_at': datetime.now().isoformat(),
        }
        
        with TASK_QUEUE_LOCK:
            TASK_QUEUE.append(task)
        
        return jsonify({
            'success': True,
            'task': task
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    with TASK_QUEUE_LOCK:
        return jsonify(TASK_QUEUE)

@app.route('/api/tasks/<task_id>', methods=['GET'])
def get_task(task_id):
    with TASK_QUEUE_LOCK:
        for task in TASK_QUEUE:
            if task['id'] == task_id:
                return jsonify(task)
    return jsonify({'error': '任务不存在'}), 404

@app.route('/api/history', methods=['GET'])
def get_history():
    with HISTORY_LOCK:
        return jsonify(HISTORY)

@app.route('/api/history/<task_id>', methods=['DELETE'])
def delete_history_item(task_id):
    global HISTORY
    with HISTORY_LOCK:
        HISTORY = [h for h in HISTORY if h['id'] != task_id]
        save_history()
        return jsonify({'success': True})

@app.route('/api/history', methods=['DELETE'])
def clear_history():
    global HISTORY
    with HISTORY_LOCK:
        for item in HISTORY:
            audio_file = item.get('audio_file')
            if audio_file:
                audio_path = os.path.join(AUDIO_DIR, audio_file)
                if os.path.exists(audio_path):
                    os.remove(audio_path)
        HISTORY = []
        save_history()
        return jsonify({'success': True})

@app.route('/api/audio/<filename>', methods=['GET'])
def get_audio(filename):
    audio_path = os.path.join(AUDIO_DIR, secure_filename(filename))
    if os.path.exists(audio_path):
        return send_file(audio_path, mimetype='audio/wav')
    return jsonify({'error': '音频文件不存在'}), 404

@app.route('/api/status', methods=['GET'])
def get_status():
    model_info = {}
    with MODELS_LOCK:
        for model_id, model in MODELS.items():
            model_info[model_id] = 'loaded'
    
    pending_count = 0
    processing_count = 0
    completed_count = 0
    with TASK_QUEUE_LOCK:
        for task in TASK_QUEUE:
            if task['status'] == 'pending':
                pending_count += 1
            elif task['status'] == 'processing':
                processing_count += 1
            elif task['status'] == 'completed':
                completed_count += 1
    
    return jsonify({
        'loaded_models': model_info,
        'queue_stats': {
            'pending': pending_count,
            'processing': processing_count,
            'completed': completed_count,
            'total': len(TASK_QUEUE)
        },
        'history_count': len(HISTORY)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
