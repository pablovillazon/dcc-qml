import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Try GPU device; fall back to CPU if GPU setup fails
try:
    n_qubits = 2  # Two qubits for the simple 2D problem
    dev = qml.device("lightning.gpu", wires=n_qubits)  # CUDA-Q-compatible GPU backend
    print("****** Using GPU-accelerated device: lightning.gpu  *****")
except Exception as e:
    print(f"/////GPU device failed: {e}. Falling back to CPU device./////")
    dev = qml.device("default.qubit", wires=n_qubits)  # CPU fallback

# Define the variational quantum circuit
n_layers = 2  # Number of variational layers
@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    # Encode input data (2D points) into qubit rotations
    qml.RX(inputs[0], wires=0)  # Feature 1 -> rotation on qubit 0
    qml.RY(inputs[1], wires=1)  # Feature 2 -> rotation on qubit 1
    
    # Variational ansatz: a simple layered circuit
    for layer in range(n_layers):
        
        # Parameterized rotations
        qml.RZ(weights[layer, 0], wires=0)
        qml.RY(weights[layer, 1], wires=1)

        # Entangling layer (Entrelazamiento, operacion cuantica CNOT)
        qml.CNOT(wires=[0, 1])
    
    # Measure the expectation value of Pauli-Z on qubit 0 as the output
    return qml.expval(qml.PauliZ(0))

# Generate a simple toy dataset (four classes in 2D)
n_samples = 100  # Increase to 100 samples for better visualization
X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2, 
                          n_redundant=0, n_clusters_per_class=1, random_state=42)
y = 2 * y - 1  # Convert labels to [-1, 1] for consistency with quantum output

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the cost function
def cost(weights, X, y):
    predictions = [quantum_circuit(x, weights) for x in X]
    pred = np.sign(predictions)  # Map to -1 or 1 for binary classification
    return np.mean((pred - y) ** 2)  # Mean squared error

# Initialize weights and optimize
np.random.seed(42)
weights = np.random.randn(n_layers, 2) * 0.1  # Small random initial weights

opt = qml.AdamOptimizer(stepsize=0.1)
steps = 50  # Number of training steps
cost_history = []

for i in range(steps):
    weights = opt.step(lambda w: cost(w, X_train, y_train), weights)
    current_cost = cost(weights, X_train, y_train)
    cost_history.append(current_cost)
    if (i + 1) % 10 == 0:
        print(f"Step {i+1}, Cost: {current_cost:.4f}")

# Evaluate on test set
test_predictions = np.sign([quantum_circuit(x, weights) for x in X_test])
accuracy = np.mean(test_predictions == y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Enhanced visualization: Combine dataset, cost history, and decision boundary
plt.figure(figsize=(15, 5))

# Plot the dataset
plt.subplot(1, 3, 1)
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.6)
plt.colorbar(scatter, label='Class (-1 or 1)')
plt.title("Dataset: 2D Points with Classes")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Plot the cost history
plt.subplot(1, 3, 2)
plt.plot(cost_history, 'b-', label="Training Cost")
plt.title("Training Cost Over Steps")
plt.xlabel("Step")
plt.ylabel("Cost")
plt.legend()

# Plot the decision boundary
plt.subplot(1, 3, 3)
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
Z = np.array([quantum_circuit(np.array([x, y]), weights) for x, y in np.c_[xx.ravel(), yy.ravel()]])
Z = np.sign(Z).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4, cmap='bwr')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.6)
plt.title("Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.tight_layout()
plt.show()