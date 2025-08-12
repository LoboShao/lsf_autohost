import torch
import torch.nn as nn
import numpy as np

class HostSelectorMLP(nn.Module):
    def __init__(self, max_num_hosts):
        super().__init__()
        self.max_num_hosts = max_num_hosts
        self.input_dim = 2 + 2 * max_num_hosts  # 2 job features + 2*num_hosts host features
        self.hidden_dim = 32
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, max_num_hosts)
        )

    def forward(self, input_vec, num_hosts):
        """
        input_vec: [input_dim] tensor (padded if num_hosts < max_num_hosts)
        num_hosts: int
        Returns: [num_hosts] tensor (priority scores)
        """
        out = self.model(input_vec)
        return out[:num_hosts]

def load_model(model_path, max_num_hosts=10):
    model = HostSelectorMLP(max_num_hosts)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def select_host_priorities(model, input_vector):
    """
    input_vector: list or array of shape [2 + 2*num_hosts]
    Returns: scores list of shape [num_hosts]
    """
    input_tensor = torch.tensor(input_vector, dtype=torch.float32)
    num_hosts = (len(input_vector) - 2) // 2
    with torch.no_grad():
        scores = model(input_tensor, num_hosts).numpy()
    return scores.tolist()

if __name__ == "__main__":
    max_num_hosts = 10
    model = HostSelectorMLP(max_num_hosts)
    print(model)

    # Example input: 4 hosts
    job = [0.5, 0.8]
    hosts = [0.1, 0.2, 0.5, 0.1, 0.7, 0.8, 0.2, 0.9]  # 4 hosts, 2 features each
    input_vec = job + hosts + [0.0] * (2 * (max_num_hosts - 4))  # pad to max_num_hosts
    scores = select_host_priorities(model, input_vec)
    print("Priority scores:", scores)

    # Save model
    torch.save(model.state_dict(), 'model.pt')
