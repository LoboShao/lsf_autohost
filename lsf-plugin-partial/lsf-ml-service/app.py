from flask import Flask, request, jsonify
from model import load_model, select_host_priorities

app = Flask(__name__)

MAX_NUM_HOSTS = 10

model = load_model('model.pt', max_num_hosts=MAX_NUM_HOSTS)

@app.route('/select_host', methods=['POST'])
def select_host():
    data = request.get_json()
    input_vector = data['input_vector']  # Flat vector: [job_features, host_features...]
    print(f"DEBUG: Received input vector: {input_vector}")
    scores = select_host_priorities(model, input_vector)
    return jsonify({'scores': [round(s, 4) for s in scores]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
