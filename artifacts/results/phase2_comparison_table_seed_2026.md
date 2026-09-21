# Phase 2 Model Comparison Table (Seed 2026)

| Configuration | Macro AUROC | 95% Bootstrap CI | Balanced Acc. | 95% Bootstrap CI |
| :--- | :---: | :---: | :---: | :---: |
| Metadata only (Age + Sex) | 0.6026 | [0.496, 0.721] | 0.2302 | [0.139, 0.327] |
| UNI2-h alone (no metadata) | 0.8952 | [0.816, 0.958] | 0.6063 | [0.458, 0.776] |
| UNI2-h alone + Metadata | 0.8885 | [0.806, 0.952] | 0.6063 | [0.458, 0.776] |
| Virchow2 alone (no metadata) | 0.8737 | [0.783, 0.947] | 0.5389 | [0.387, 0.709] |
| Virchow2 alone + Metadata | 0.8759 | [0.784, 0.949] | 0.5389 | [0.387, 0.709] |
| Prov-GigaPath alone (no metadata) | 0.8827 | [0.798, 0.948] | 0.5135 | [0.352, 0.699] |
| Prov-GigaPath alone + Metadata | 0.8747 | [0.792, 0.941] | 0.5183 | [0.357, 0.692] |
| Best Single Model + Metadata | 0.8885 | [0.806, 0.952] | 0.6063 | [0.458, 0.776] |
| Fused (Agent-Selected) (no metadata) | 0.8904 | [0.810, 0.954] | 0.5718 | [0.410, 0.745] |
| **Fused (Agent-Selected) + Metadata** | **0.8907** | **[0.809, 0.955]** | **0.5877** | **[0.428, 0.762]** |
| Prism2 VLM alone (no metadata) | 0.9191 | [0.865, 0.960] | 0.4877 | [0.332, 0.639] |
| Prism2 VLM alone + Metadata | 0.9136 | [0.864, 0.955] | 0.5143 | [0.362, 0.670] |
| Quad Multimodal Late Fusion (+ Prism2 VLM) (no metadata) | 0.9122 | [0.847, 0.965] | 0.6353 | [0.466, 0.810] |
| **Quad Multimodal Late Fusion (+ Prism2 VLM) + Metadata (Champion)** | **0.9090** | **[0.845, 0.960]** | **0.6353** | **[0.466, 0.810]** |
