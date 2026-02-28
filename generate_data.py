import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random

# =======================
# 1. Data Generation
# =======================

N = 256
SAMPLES = 8000

def gen_normal():
    t = np.linspace(0,1,N)
    return np.sin(2*np.pi*5*t) + 0.05*np.random.randn(N)

def gen_anomaly():
    x = gen_normal()
    if random.random()<0.5:
        idx = random.randint(10,N-10)
        x[idx:idx+5]+=np.random.uniform(2,4)
    else:
        x += np.linspace(0,3,N)
    return x

X=[]
y=[]
for _ in range(SAMPLES):
    if random.random()<0.5:
        X.append(gen_normal())
        y.append(0)
    else:
        X.append(gen_anomaly())
        y.append(1)

X=np.array(X,dtype=np.float32)
y=np.array(y,dtype=np.float32)

X=torch.tensor(X).unsqueeze(1)
y=torch.tensor(y).unsqueeze(1)

# =======================
# 2. 1D CNN Model
# =======================

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv1d(1,8,7),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8,16,5),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc=nn.Linear(16,1)

    def forward(self,x):
        x=self.net(x)
        x=x.view(x.size(0),-1)
        return torch.sigmoid(self.fc(x))

model=Net()
opt=torch.optim.Adam(model.parameters(),lr=1e-3)
loss_fn=nn.BCELoss()

# =======================
# 3. Training
# =======================

for epoch in range(6):
    pred=model(X)
    loss=loss_fn(pred,y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print("epoch",epoch,"loss",loss.item())

# =======================
# 4. System Simulation Parameters
# =======================

QMAX=50
PROCESS_RATE=1
steps=4000

def adaptive_T(q):
    if q<0.3*QMAX: return 0.5
    elif q<0.7*QMAX: return 0.7
    else: return 0.85

# =======================
# 5. Simulation Function
# =======================

def simulate(arrival_rate,adaptive=True):
    queue=0
    latencies=[]
    acc=[]
    for t in range(steps):

        queue+=arrival_rate

        if queue>QMAX:
            queue=QMAX

        if queue>=1:
            queue-=PROCESS_RATE

            idx=random.randint(0,len(X)-1)
            x=X[idx:idx+1]
            label=y[idx].item()

            score=model(x).item()

            T=adaptive_T(queue) if adaptive else 0.7

            pred=int(score>T)
            acc.append(pred==label)

            latencies.append(queue)

    return np.mean(latencies), np.percentile(latencies,99), np.mean(acc)

# =======================
# 6. Stress Test and Data Collection
# =======================

loads=np.linspace(0.2,2,15)

fixed_avg=[]
fixed_p99=[]
fixed_acc=[]

adp_avg=[]
adp_p99=[]
adp_acc=[]

for l in loads:
    a,b,c=simulate(l,False)
    fixed_avg.append(a)
    fixed_p99.append(b)
    fixed_acc.append(c)

    a,b,c=simulate(l,True)
    adp_avg.append(a)
    adp_p99.append(b)
    adp_acc.append(c)

# =======================
# 7. Graphing
# =======================

plt.figure()
plt.plot(loads,fixed_p99,label="Fixed")
plt.plot(loads,adp_p99,label="Adaptive")
plt.title("P99 Latency")
plt.legend()

plt.figure()
plt.plot(loads,fixed_acc,label="Fixed")
plt.plot(loads,adp_acc,label="Adaptive")
plt.title("Accuracy")
plt.legend()

plt.show()