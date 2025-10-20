import torch
from flask import Flask, request, jsonify
import sys
from pathlib import Path

# Add parent directory to path to import from training
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.attention_scheduler_model import AttentionSchedulerPolicy

app = Flask(__name__)

MAX_NUM_HOSTS = 30

# Helper functions for model loading and inference
def load_model(model_path='checkpoint.pt', num_hosts=30):
    """Load the AttentionSchedulerPolicy model from training"""
    # Calculate obs_dim based on expected input format
    # host_features[2*num_hosts] + job_features[4] + queue_features[4]
    obs_dim = 2 * num_hosts + 8
    action_dim = num_hosts

    model = AttentionSchedulerPolicy(obs_dim, action_dim, num_hosts)

    if model_path != 'dummy_path':
        # Load checkpoint - handle both state_dict and full checkpoint formats
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        if isinstance(checkpoint, dict) and 'policy_state_dict' in checkpoint:
            # Full training checkpoint format
            model.load_state_dict(checkpoint['policy_state_dict'])
        else:
            # Direct state_dict format
            model.load_state_dict(checkpoint)

    model.eval()
    return model

@torch.no_grad()
def select_host_priorities(model, input_vector):
    """
    Get host priorities from model

    input_vector format: [host1_avail_cores_norm, host1_avail_mem_norm, ...,
                          hostN_avail_cores_norm, hostN_avail_mem_norm,
                          job_cores_norm, job_mem_norm, job_duration_norm, is_deferred,
                          batch_progress, queue_pressure, core_pressure, memory_pressure]
    Returns: scores list of shape [num_hosts]
    """
    model.eval()
    input_tensor = torch.tensor(input_vector, dtype=torch.float32)
    action_mean, _, _ = model(input_tensor, deterministic=True)
    return action_mean.numpy().tolist()

# Load the AttentionSchedulerPolicy model from training
model = load_model('best_model.pt', num_hosts=MAX_NUM_HOSTS)

# Counter for inference requests
inference_count = 0

@app.route('/select_host', methods=['POST'])
def select_host():
    global inference_count
    inference_count += 1

    data = request.get_json()
    # Input format: [host_features(2*n), job_features(4), queue_features(4)]
    input_vector = data['input_vector']

    scores = select_host_priorities(model, input_vector)

    # Print vector length on first request
    if inference_count == 1:
        num_hosts = (len(input_vector) - 8) // 2
        print(f"\nReceived input vector with {num_hosts} hosts")
        print(f"Vector length: {len(input_vector)} = {num_hosts}*2 + 8\n")

    # Print count every 100 requests
    if inference_count % 100 == 0:
        print(f"\nTotal inferences: {inference_count}\n")

    return jsonify({'scores': [round(s, 4) for s in scores]})

@app.route('/stats', methods=['GET'])
def get_stats():
    return jsonify({'inference_count': inference_count})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
