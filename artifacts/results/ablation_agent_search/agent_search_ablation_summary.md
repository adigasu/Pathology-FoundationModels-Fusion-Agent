# Autonomous Agent Search Architecture Ablation Study

Cross-version comparison across **v1 (Sequential Agent)**, **v2 (Exploitation Agent)**, **v3 (Hierarchical Agent)**, **v4 (Autonomous Agent)**, and **v5 (Unified Agent)**.

### Statistical Protocol:
- **Development Cohort**: $N=163$ patients, 5-Fold Stratified Cross-Validation
- **Held-out Test Cohort**: $N=41$ patients, evaluated strictly once per seed
- **Repeated Seeds**: 42, 1337, 2026
- **Uncertainty Quantification**: Bessel's correction ($s$, $\text{ddof}=1$), 1,000 patient-level bootstrap 95% CIs
- **Prior-Adjusted Balanced Accuracy**: Power calibration ($\alpha_{\text{prior}}=0.4$)

## Agent Search Ablation Comparison Table

| Agent Version | Search Paradigm / Optimization Policy | Avg Trials | Dev CV AUROC (Mean ± SD) | Discovered Champion Architecture | Tri-Vision Test Macro AUROC (ddof=1) | 95% Bootstrap CI (Seed 42) | Test Bal. Acc (Raw $\to$ Adj $\alpha=0.4$) | Quad-Model Test Macro AUROC | Search Time (s) |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| v1: Sequential Agent | Fixed sequential exploration (T01-T12) across... | 21 | 0.8303 ± 0.0099 | Late Temperature ($\tau=1.5$) [CONCAT] | 0.9006 ± 0.0109 | [0.834, 0.952] | 55.1% $\to$ 59.1% | 0.9090 ± 0.0122 | 96.7s |
| **v2: Exploitation Agent** | Representation-capacity-first exploitation: e... | 19 | 0.8297 ± 0.0107 | Late Temperature ($\tau=1.5$) [CONCAT] | **0.9030 ± 0.0149** | [0.834, 0.952] | 56.5% $\to$ 59.3% | **0.9082 ± 0.0111** | 87.6s |
| v3: Hierarchical Agent | Decoupled 3-stage search: Stage 1 isolates an... | 16 | 0.8332 ± 0.0024 | Late Temperature ($\tau=1.2$) [MAX] | 0.8981 ± 0.0147 | [0.824, 0.943] | 61.2% $\to$ 60.1% | 0.9045 ± 0.0186 | 77.5s |
| **v4: Autonomous Agent** | Hypothesis-driven proposal loop over 25 trial... | 9 | 0.8317 ± 0.0088 | Late Temperature ($\tau=1.5$) [CONCAT] | **0.9008 ± 0.0119** | [0.837, 0.952] | 54.9% $\to$ 57.9% | **0.9101 ± 0.0078** | 64.7s |
| v5: Unified Agent | 3-stage unified pipeline: Stage 1 screens sta... | 13 | 0.8341 ± 0.0013 | Late Temperature ($\tau=1.2$) [MAX] | 0.8963 ± 0.0141 | [0.824, 0.943] | 60.3% $\to$ 61.3% | 0.9035 ± 0.0175 | 78.4s |

### Multi-Seed Breakdown (Seeds 42 / 1337 / 2026)

| Agent Version | Seed 42 Tri-Vision AUROC | Seed 1337 Tri-Vision AUROC | Seed 2026 Tri-Vision AUROC | Seed 42 Quad AUROC | Seed 1337 Quad AUROC | Seed 2026 Quad AUROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **v1: Sequential Agent** | 0.8988 | 0.9123 | 0.8907 | 0.8969 | 0.9213 | 0.9087 |
| **v2: Exploitation Agent** | 0.8988 | 0.9195 | 0.8907 | 0.8969 | 0.9190 | 0.9087 |
| **v3: Hierarchical Agent** | 0.8849 | 0.9139 | 0.8956 | 0.8833 | 0.9184 | 0.9116 |
| **v4: Autonomous Agent** | 0.8979 | 0.9139 | 0.8907 | 0.9030 | 0.9184 | 0.9087 |
| **v5: Unified Agent** | 0.8849 | 0.9120 | 0.8920 | 0.8833 | 0.9153 | 0.9117 |
