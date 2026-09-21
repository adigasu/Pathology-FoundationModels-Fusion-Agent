# Phase 2 Model Comparison Table (Seed 1337)

| Configuration | Macro AUROC | 95% Bootstrap CI | Balanced Acc. | 95% Bootstrap CI |
| :--- | :---: | :---: | :---: | :---: |
| Metadata only (Age + Sex) | 0.6959 | [0.582, 0.799] | 0.2321 | [0.146, 0.335] |
| UNI2-h alone (no metadata) | 0.9061 | [0.857, 0.954] | 0.5429 | [0.411, 0.694] |
| UNI2-h alone + Metadata | 0.9092 | [0.861, 0.955] | 0.5262 | [0.380, 0.698] |
| Virchow2 alone (no metadata) | 0.8905 | [0.827, 0.947] | 0.5480 | [0.381, 0.722] |
| Virchow2 alone + Metadata | 0.8930 | [0.830, 0.950] | 0.5373 | [0.363, 0.707] |
| Prov-GigaPath alone (no metadata) | 0.9166 | [0.860, 0.965] | 0.5845 | [0.438, 0.750] |
| Prov-GigaPath alone + Metadata | 0.9146 | [0.860, 0.964] | 0.5845 | [0.438, 0.750] |
| Best Single Model + Metadata | 0.9146 | [0.860, 0.964] | 0.5845 | [0.438, 0.750] |
| Fused (Agent-Selected) (no metadata) | 0.9105 | [0.856, 0.960] | 0.5587 | [0.416, 0.730] |
| **Fused (Agent-Selected) + Metadata** | **0.9139** | **[0.860, 0.963]** | **0.5587** | **[0.403, 0.729]** |
| Prism2 VLM alone (no metadata) | 0.8774 | [0.800, 0.939] | 0.5567 | [0.391, 0.719] |
| Prism2 VLM alone + Metadata | 0.8826 | [0.809, 0.944] | 0.5746 | [0.410, 0.747] |
| Quad Multimodal Late Fusion (+ Prism2 VLM) (no metadata) | 0.9173 | [0.860, 0.965] | 0.6183 | [0.443, 0.786] |
| **Quad Multimodal Late Fusion (+ Prism2 VLM) + Metadata (Champion)** | **0.9184** | **[0.863, 0.965]** | **0.5825** | **[0.427, 0.748]** |
