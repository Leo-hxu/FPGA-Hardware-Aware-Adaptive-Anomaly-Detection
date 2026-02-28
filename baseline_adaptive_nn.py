import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# =======================
# 1. Reproducibility
# =======================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =======================
# 2. Data Generation
# =======================

N = 256
SAMPLES = 8000


def gen_normal():
    t = np.linspace(0, 1, N)
    return np.sin(2 * np.pi * 5 * t) + 0.05 * np.random.randn(N)


def gen_anomaly():
    x = gen_normal()
    if random.random() < 0.5:
        idx = random.randint(10, N - 10)
        x[idx:idx + 5] += np.random.uniform(2, 4)
    else:
        x += np.linspace(0, 3, N)
    return x


X = []
y = []
for _ in range(SAMPLES):
    if random.random() < 0.5:
        X.append(gen_normal())
        y.append(0)
    else:
        X.append(gen_anomaly())
        y.append(1)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

X = torch.tensor(X).unsqueeze(1)
y = torch.tensor(y).unsqueeze(1)

# =======================
# 3. 1D CNN Model
# =======================


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 8, 7),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, 5),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))


model = Net()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

# =======================
# 4. Training
# =======================

for epoch in range(6):
    pred = model(X)
    loss = loss_fn(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print("epoch", epoch, "loss", loss.item())

# =======================
# 5. System Simulation Parameters
# =======================

QMAX = 50
PROCESS_RATE = 1
STEPS = 4000


def adaptive_T(q):
    if q < 0.3 * QMAX:
        return 0.5
    if q < 0.7 * QMAX:
        return 0.7
    return 0.85


# =======================
# 6. Simulation Function
# =======================

def simulate(arrival_rate, adaptive=True):
    queue = 0
    latencies = []
    acc = []
    
    # Base processing capacity (1 request per step)
    base_process_rate = 1.0
    
    for _ in range(STEPS):
        queue += arrival_rate

        if queue > QMAX:
            queue = QMAX

        if queue >= 1:
            # Core logic fix: Threshold affects how fast we can process
            # Lower threshold means stricter check = slower (normal)
            # Higher threshold means looser check = faster (shortcut taken)
            threshold = adaptive_T(queue) if adaptive else 0.7
            
            # Simulated processing time based on threshold
            # If threshold is 0.85 (loose), we process 1.5x faster
            # If threshold is 0.5 (strict), we process 0.8x faster
            if threshold > 0.8:
                actual_process_rate = base_process_rate * 1.5
            elif threshold < 0.6:
                actual_process_rate = base_process_rate * 0.8
            else:
                actual_process_rate = base_process_rate
                
            queue -= min(queue, actual_process_rate)

            # Sample random data just to record accuracy penalty
            idx = random.randint(0, len(X) - 1)
            x = X[idx:idx + 1]
            label = y[idx].item()

            score = model(x).item()

            pred = int(score > threshold)
            acc.append(pred == label)

            latencies.append(queue)
        else:
            latencies.append(0)

    return np.mean(latencies), np.percentile(latencies, 99), np.mean(acc)


# =======================
# 7. Stress Test and Data Collection
# =======================

loads = np.linspace(0.2, 2, 15)

fixed_p99 = []
fixed_acc = []

adp_p99 = []
adp_acc = []

for load in loads:
    _, p99, acc = simulate(load, False)
    fixed_p99.append(p99)
    fixed_acc.append(acc)

    _, p99, acc = simulate(load, True)
    adp_p99.append(p99)
    adp_acc.append(acc)

# =======================
# 8. Graphing and Saving
# =======================

plt.figure()
plt.plot(loads, fixed_p99, label="Fixed")
plt.plot(loads, adp_p99, label="Adaptive")
plt.title("P99 Latency")
plt.xlabel("Load")
plt.ylabel("Queue depth")
plt.legend()
plt.tight_layout()
plt.savefig("p99_latency.png", dpi=150)
plt.close()
print("saved p99_latency.png")

plt.figure()
plt.plot(loads, fixed_acc, label="Fixed")
plt.plot(loads, adp_acc, label="Adaptive")
plt.title("Accuracy")
plt.xlabel("Load")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("accuracy.png", dpi=150)
plt.close()
print("saved accuracy.png")
