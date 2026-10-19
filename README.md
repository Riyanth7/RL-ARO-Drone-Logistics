# RL-ARO Drone Logistics Simulation (Pure Python)
5-Member Python implementation and statistical reproduction of *"AI-driven computing framework for intelligent scheduling and route optimization in large-scale IoT-enabled drone logistics networks"* (Yu Qian, 2026).
![RL-ARO Simulation Dashboard](RL_ARO_Simulation_Dashboard.png)
## System Architecture & 5-Member Role Allocation

| File | Owner | Core Responsibilities & Equations |
| :--- | :--- | :--- |
| **`agent.py`** | **Member 1 (ML & DL)** | RL-ARO Agent, Multi-Head Attention State Encoding (Algorithm 1), Experience Replay, and Reward Function (**Eq. 1**). |
| **`network.py`** | **Member 2 (Networking & IoT)** | IEEE 802.11 Wireless Link Simulator, Sensor Outlier Filtering, Edge-Cloud Latency, and Blockchain Delay (**Eq. 3, 5–12**). |
| **`environment.py`** | **Member 3 (Routing & Physics)** | 1000m x 1000m 10-Node Healthcare Airspace Graph, Drone Battery/Speed Physics (**Eq. 2 & 4**), and Baselines (Dijkstra, ACO, TSP-D, DBSCAN). |
| **`evaluator.py`** | **Member 4 (Data & Validation)** | 5-Seed Statistical Evaluator, Paired Student t-test, Performance Metrics (**Eq. 13–20**), and Automated Chart Generator. |
| **`main.py`** | **Member 5 (Integration Lead)** | Master CTDE Training & Multi-Seed Evaluation Orchestrator connecting all 4 modules. |

## Locked Assumptions Register
* **Random Seeds:** `42, 123, 256, 512, 1024` (5 independent simulation runs)
* **Airspace & Fleet:** `1000m x 1000m` 3D domain, `20 UAV` fleet scaling, `1.5 kg` medical payload
* **Reward & Cost Weights:** Alpha = `0.55`, Beta = `0.45`, Lambda = `0.50`
## How to Run
1. Install required Python libraries:
   `pip install numpy matplotlib networkx scipy`
2. Run the master simulation script:
   `python main.py`
