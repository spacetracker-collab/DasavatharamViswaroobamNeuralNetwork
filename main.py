import numpy as np

# -----------------------------
# Utility functions
# -----------------------------
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

# -----------------------------
# Vishwaroopam Model
# Sequential Neural Network
# -----------------------------
class VishwaroopamNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)

    def forward(self, x):
        self.h = relu(np.dot(x, self.W1))
        self.out = softmax(np.dot(self.h, self.W2))
        return self.out

# -----------------------------
# Dasavatharam Model
# Multi-Agent Ensemble Network
# -----------------------------
class Agent:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)

    def forward(self, x):
        h = relu(np.dot(x, self.W1))
        out = softmax(np.dot(h, self.W2))
        return out


class DasavatharamNN:
    def __init__(self, num_agents, input_size, hidden_size, output_size):
        self.agents = [
            Agent(input_size, hidden_size, output_size)
            for _ in range(num_agents)
        ]

    def forward(self, x):
        outputs = []
        for agent in self.agents:
            outputs.append(agent.forward(x))

        # Ensemble averaging
        final_output = np.mean(outputs, axis=0)
        return final_output


# -----------------------------
# Simulation
# -----------------------------
if __name__ == "__main__":
    np.random.seed(42)

    input_size = 5
    hidden_size = 10
    output_size = 3

    # Sample input
    x = np.random.randn(input_size)

    # Vishwaroopam Model
    v_model = VishwaroopamNN(input_size, hidden_size, output_size)
    v_output = v_model.forward(x)

    print("Vishwaroopam Output (Sequential NN):")
    print(v_output)

    # Dasavatharam Model (10 agents)
    d_model = DasavatharamNN(
        num_agents=10,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size
    )
    d_output = d_model.forward(x)

    print("\nDasavatharam Output (Multi-Agent Ensemble):")
    print(d_output)
