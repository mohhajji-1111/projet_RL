"""
🌐 Web Dashboard Backend - Flask API
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
training_state = {
    'is_training': False,
    'current_episode': 0,
    'total_episodes': 0,
    'current_reward': 0,
    'best_reward': float('-inf'),
}

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)


# ==================== API ENDPOINTS ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})


@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start training."""
    config = request.json
    
    if training_state['is_training']:
        return jsonify({'error': 'Training already in progress'}), 400
    
    training_state['is_training'] = True
    training_state['total_episodes'] = config.get('episodes', 1000)
    training_state['current_episode'] = 0
    
    # TODO: Start actual training in background
    
    return jsonify({
        'status': 'started',
        'config': config
    })


@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Stop training."""
    training_state['is_training'] = False
    
    return jsonify({'status': 'stopped'})


@app.route('/api/training/status', methods=['GET'])
def get_training_status():
    """Get current training status."""
    return jsonify(training_state)


@app.route('/api/training/logs', methods=['GET'])
def get_training_logs():
    """Get training logs."""
    # TODO: Read from log file
    logs = [
        "Episode 1: Reward=-100.5",
        "Episode 2: Reward=-85.3",
        "Episode 3: Reward=-70.2"
    ]
    return jsonify({'logs': logs})


@app.route('/api/models', methods=['GET'])
def list_models():
    """List all saved models."""
    models = []
    for model_file in models_dir.glob("*.pt"):
        stat = model_file.stat()
        models.append({
            'id': model_file.stem,
            'name': model_file.name,
            'size': stat.st_size,
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
        })
    return jsonify({'models': models})


@app.route('/api/models/<model_id>', methods=['GET'])
def get_model(model_id):
    """Get model details."""
    model_path = models_dir / f"{model_id}.pt"
    
    if not model_path.exists():
        return jsonify({'error': 'Model not found'}), 404
    
    # TODO: Load model and extract info
    return jsonify({
        'id': model_id,
        'name': f"{model_id}.pt",
        'architecture': 'DQN',
        'parameters': 1000000
    })


@app.route('/api/models/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    """Delete a model."""
    model_path = models_dir / f"{model_id}.pt"
    
    if not model_path.exists():
        return jsonify({'error': 'Model not found'}), 404
    
    model_path.unlink()
    return jsonify({'status': 'deleted'})


@app.route('/api/models/upload', methods=['POST'])
def upload_model():
    """Upload a model."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filepath = models_dir / file.filename
    file.save(filepath)
    
    return jsonify({
        'status': 'uploaded',
        'filename': file.filename
    })


@app.route('/api/evaluation/run', methods=['POST'])
def run_evaluation():
    """Run evaluation."""
    config = request.json
    model_id = config.get('model_id')
    num_episodes = config.get('num_episodes', 10)
    
    # TODO: Run actual evaluation
    
    return jsonify({
        'status': 'completed',
        'results': {
            'average_reward': 100.5,
            'success_rate': 0.8,
            'episodes': num_episodes
        }
    })


@app.route('/api/evaluation/results', methods=['GET'])
def get_evaluation_results():
    """Get evaluation results."""
    # TODO: Load from database
    results = {
        'average_reward': 100.5,
        'success_rate': 0.8,
        'total_episodes': 50
    }
    return jsonify(results)


@app.route('/api/metrics/history', methods=['GET'])
def get_metrics_history():
    """Get metrics history."""
    # TODO: Load from database
    history = {
        'episodes': list(range(1, 101)),
        'rewards': np.random.randn(100).tolist(),
        'success_rate': np.random.rand(100).tolist()
    }
    return jsonify(history)


@app.route('/api/visualization/trajectory', methods=['GET'])
def get_trajectory():
    """Get trajectory data."""
    # TODO: Get actual trajectory
    trajectory = {
        'points': [[100, 100], [150, 120], [200, 150]],
        'obstacles': [[300, 300, 50, 50]],
        'goal': [700, 500]
    }
    return jsonify(trajectory)


@app.route('/api/export/report', methods=['GET'])
def export_report():
    """Export training report."""
    # TODO: Generate report
    report = {
        'training_summary': 'Training completed successfully',
        'total_episodes': 1000,
        'best_reward': 150.5
    }
    
    # Save to file
    report_path = Path("reports") / "report.json"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return send_file(report_path, as_attachment=True)


# ==================== WEBSOCKET EVENTS ====================

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print('Client connected')
    emit('connected', {'data': 'Connected to server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print('Client disconnected')


@socketio.on('subscribe_metrics')
def handle_subscribe_metrics():
    """Subscribe to live metrics."""
    # TODO: Start sending live metrics
    emit('metrics_update', {
        'episode': 1,
        'reward': 100.5,
        'steps': 500
    })


# ==================== BACKGROUND TASKS ====================

def send_live_metrics():
    """Send live metrics to connected clients."""
    while training_state['is_training']:
        # TODO: Get actual metrics
        socketio.emit('metrics_update', {
            'episode': training_state['current_episode'],
            'reward': training_state['current_reward'],
            'timestamp': datetime.now().isoformat()
        })
        socketio.sleep(1)


if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    print("📡 API available at: http://localhost:5000")
    print("📊 Dashboard at: http://localhost:3000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
