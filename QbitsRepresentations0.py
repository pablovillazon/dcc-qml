import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 1. Set up the GPU-accelerated device (requires PennyLane-Lightning[GPU] and CUDA-Q)
n_qubits = 4  # Increase qubits for more complex data (2 qubits per feature for 2D)
dev = qml.device("lightning.gpu", wires=n_qubits)  # CUDA-Q-compatible GPU backend

# 2. Define the variational quantum circuit
n_layers = 3  # Increase layers for more expressive model
def quantum_circuit(inputs, weights):
    # Encode 2D input data into qubit rotations (using 2 qubits per feature)
    qml.RX(inputs[0], wires=0)  # Feature 1 -> qubit 0
    qml.RY(inputs[0], wires=1)  # Feature 1 -> qubit 1 (redundancy for robustness)
    qml.RX(inputs[1], wires=2)  # Feature 2 -> qubit 2
    qml.RY(inputs[1], wires=3)  # Feature 2 -> qubit 3
    
    # Variational ansatz: layered circuit with more entangling
    for layer in range(n_layers):
        # Parameterized rotations on all qubits
        for i in range(n_qubits):
            qml.RZ(weights[layer, i, 0], wires=i)
            qml.RY(weights[layer, i, 1], wires=i)
        # Entangling layers (more connections for richer entanglement)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[1, 2])  # Additional entanglement for 4 qubits
    
    # Measure expectation values of Pauli-Z on qubits 0 and 2 for binary output
    return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(2))]

# 3. Create a larger, more complex dataset
n_samples = 100  # Increase to 100 samples for better visualization
X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2, 
                          n_redundant=0, n_clusters_per_class=1, random_state=42)
y = 2 * y - 1  # Convert labels to [-1, 1] for consistency with quantum output

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Define the cost function (mean squared error for binary classification)
@qml.qnode(dev)
def qnode(inputs, weights):
    return quantum_circuit(inputs, weights)

def cost(weights, X, y):
    predictions = np.array([qnode(x, weights) for x in X])
    # Combine the two PauliZ measurements into a single prediction (-1 or 1)
    pred = np.sign(predictions[:, 0] + predictions[:, 1])  # Aggregate measurements
    return np.mean((pred - y) ** 2)

# 5. Initialize weights and optimize
np.random.seed(42)
weights_shape = (n_layers, n_qubits, 2)  # Shape for weights: (layers, qubits, parameters per qubit)
weights = np.random.randn(*weights_shape) * 0.1  # Small random initial weights

opt = qml.AdamOptimizer(stepsize=0.05)  # Slightly smaller stepsize for stability with more data
steps = 100  # Increase steps for convergence with larger dataset
cost_history = []

for i in range(steps):
    weights = opt.step(lambda w: cost(w, X_train, y_train), weights)
    current_cost = cost(weights, X_train, y_train)
    cost_history.append(current_cost)
    if (i + 1) % 10 == 0:
        print(f"Step {i+1}, Cost: {current_cost:.4f}")

# 6. Evaluate on test set
test_predictions = np.array([qnode(x, weights) for x in X_test])
test_pred = np.sign(test_predictions[:, 0] + test_predictions[:, 1])
accuracy = np.mean(test_pred == y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")

# 7. Enhanced visualization
plt.figure(figsize=(12, 5))

# Plot the dataset
plt.subplot(1, 2, 1)
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.6)
plt.colorbar(scatter, label='Class (-1 or 1)')
plt.title("Dataset: 2D Points with Classes")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Plot the cost history
plt.subplot(1, 2, 2)
plt.plot(cost_history, 'b-', label="Training Cost")
plt.title("Training Cost Over Steps")
plt.xlabel("Step")
plt.ylabel("Cost")
plt.legend()

plt.tight_layout()
plt.show()

# 8. Visualize decision boundary (optional, for fun!)
def plot_decision_boundary(X, y, weights):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = np.array([qnode(np.array([x, y]), weights) for x, y in np.c_[xx.ravel(), yy.ravel()]])
    Z = np.sign(Z[:, 0] + Z[:, 1]).reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='bwr')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.6)
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

plot_decision_boundary(X_train, y_train, weights)