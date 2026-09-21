# Phase 2 Model Comparison Table (Seed 42)

| Configuration | Macro AUROC | 95% Bootstrap CI | Balanced Acc. | 95% Bootstrap CI |
| :--- | :---: | :---: | :---: | :---: |
| Metadata only (Age + Sex) | 0.7084 | [0.602, 0.799] | 0.2488 | [0.177, 0.342] |
| UNI2-h alone (no metadata) | 0.8744 | [0.807, 0.932] | 0.4706 | [0.332, 0.637] |
| UNI2-h alone + Metadata | 0.8780 | [0.812, 0.936] | 0.5341 | [0.382, 0.699] |
| Virchow2 alone (no metadata) | 0.8926 | [0.827, 0.948] | 0.5115 | [0.367, 0.662] |
| Virchow2 alone + Metadata | 0.8924 | [0.827, 0.947] | 0.5294 | [0.375, 0.712] |
| Prov-GigaPath alone (no metadata) | 0.8883 | [0.817, 0.945] | 0.5056 | [0.352, 0.680] |
| Prov-GigaPath alone + Metadata | 0.8979 | [0.832, 0.950] | 0.5056 | [0.352, 0.680] |
| Best Single Model + Metadata | 0.8979 | [0.832, 0.950] | 0.5056 | [0.352, 0.680] |
| Fused (Agent-Selected) (no metadata) | 0.8959 | [0.832, 0.947] | 0.4520 | [0.314, 0.606] |
| **Fused (Agent-Selected) + Metadata** | **0.8979** | **[0.837, 0.952]** | **0.4996** | **[0.338, 0.667]** |
| Prism2 VLM alone (no metadata) | 0.8600 | [0.793, 0.924] | 0.3980 | [0.280, 0.531] |
| Prism2 VLM alone + Metadata | 0.8511 | [0.783, 0.912] | 0.4298 | [0.314, 0.554] |
| Quad Multimodal Late Fusion (+ Prism2 VLM) (no metadata) | 0.8921 | [0.837, 0.944] | 0.4579 | [0.325, 0.617] |
| **Quad Multimodal Late Fusion (+ Prism2 VLM) + Metadata (Champion)** | **0.9030** | **[0.849, 0.952]** | **0.5532** | **[0.382, 0.716]** |
