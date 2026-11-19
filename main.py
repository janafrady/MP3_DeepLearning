import ANN as N
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path
from time import perf_counter

SEED = 1
EPOCHS = 1000
LEARNING_RATE = 0.75

OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_FILE = OUTPUT_DIR / "results.csv"
PLOT_FILE = OUTPUT_DIR / "cost_comparison.png"
GRAPHS_DIR = OUTPUT_DIR / "graphs"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

# experiments = [
#     {
#         "name": "Network 1",
#         "layers": [2, 3, 2],
#         "replacement": 0
#     },
#     {
#         "name": "Network 2",
#         "layers": [2, 20, 2],
#         "replacement": 1
#     }
# ]

experiments = [
    {"name": "1 Hidden - 3 Nodes", "layers": [2, 3, 2], "replacement": 0},
    {"name": "1 Hidden - 5 Nodes", "layers": [2, 5, 2], "replacement": 0},
    {"name": "1 Hidden - 10 Nodes", "layers": [2, 10, 2], "replacement": 0},
    {"name": "1 Hidden - 20 Nodes", "layers": [2, 20, 2], "replacement": 0},
    {"name": "2 Hidden - 10 Nodes Each", "layers": [2, 10, 10, 2], "replacement": 0},
    {"name": "3 Hidden - 10 Nodes Each", "layers": [2, 10, 10, 10, 2], "replacement": 0}
]


# calcualte training accuracy
def calculate_accuracy(model, data):
    correct = 0
    total = data.xtrain.shape[1]

    for i in range(total):
        x = data.xtrain[:, i].reshape(-1, 1)

        prediction = model.predict(x)

        predicted_class = np.argmax(prediction)
        actual_class = np.argmax(data.ytrain[:, i])

        if predicted_class == actual_class:
            correct += 1

    return 100 * correct / total


# train both networks

results = []
cost_curves = []

for experiment in experiments:

    # Reset random seed for reproducibility
    np.random.seed(SEED)

    # Create original Higham dataset
    data = N.Data(10, highamdata=True)

    # Create neural network
    model = N.GeneralNetwork(
        number_of_layers=len(experiment["layers"]),
        neurons_per_layer=experiment["layers"],
        verbose=0,
        vis=False,
        activation_function="sigmoid"
    )

    # Measure training time
    start = perf_counter()

    cost = model.train(
        data,
        eta=LEARNING_RATE,
        epochs=EPOCHS,
        replacement=experiment["replacement"]
    )

    training_time = perf_counter() - start

    # Calculate final performance
    final_cost = model.cost_function(data)
    accuracy = calculate_accuracy(model, data)
    # Generate decision boundary
    model.visual(
    data,
    no_of_points=100,
    filename=str(GRAPHS_DIR / f"boundary_{'_'.join(map(str, experiment['layers']))}.png"),
    title=f"Decision Boundary: {experiment['name']}"
)
    

    # Average cost for each epoch
    avg_cost = np.mean(cost, axis=1)

    cost_curves.append({
        "name": experiment["name"],
        "cost": avg_cost
    })

    # Save numerical results
    result = {
        "network": experiment["name"],
        "architecture": str(experiment["layers"]),
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "replacement": experiment["replacement"],
        "training_time": round(training_time, 4),
        "final_cost": round(float(final_cost), 6),
        "training_accuracy": round(accuracy, 2)
    }

    results.append(result)

    # Print summary
    print(f"\n{experiment['name']}")
    print(f"Architecture: {experiment['layers']}")
    print(f"Training time: {training_time:.4f} seconds")
    print(f"Final cost: {final_cost:.6f}")
    print(f"Training accuracy: {accuracy:.2f}%")


# save results to CSV
with open(RESULTS_FILE, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)

print(f"\nResults saved to {RESULTS_FILE}")


# Plot training cost

plt.figure(figsize=(9, 5))

for curve in cost_curves:
    plt.plot(
        np.arange(1, EPOCHS + 1),
        curve["cost"],
        label=curve["name"]
    )

plt.xlabel("Epoch")
plt.ylabel("Average Training Cost")
plt.title("Neural Network Training Cost Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(PLOT_FILE, dpi=300)
plt.show()

print(f"Graph saved to {PLOT_FILE}")
