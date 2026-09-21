# Phase 2 Multi-Seed Repeatability Summary (Seeds 42, 1337, 2026)

| Configuration | Seed 42 AUROC | Seed 1337 AUROC | Seed 2026 AUROC | Mean ± SD AUROC | Mean ± SD Balanced Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Metadata only (Age + Sex) | 0.7084 | 0.6959 | 0.6026 | 0.6690 ± 0.0578 | 0.2370 ± 0.0102 |
| UNI2-h alone (no metadata) | 0.8744 | 0.9061 | 0.8952 | 0.8919 ± 0.0161 | 0.5399 ± 0.0679 |
| UNI2-h alone + Metadata | 0.8780 | 0.9092 | 0.8885 | 0.8919 ± 0.0158 | 0.5556 ± 0.0442 |
| Virchow2 alone (no metadata) | 0.8926 | 0.8905 | 0.8737 | 0.8856 ± 0.0103 | 0.5328 ± 0.0190 |
| Virchow2 alone + Metadata | 0.8924 | 0.8930 | 0.8759 | 0.8871 ± 0.0097 | 0.5352 ± 0.0051 |
| Prov-GigaPath alone (no metadata) | 0.8883 | 0.9166 | 0.8827 | 0.8959 ± 0.0182 | 0.5345 ± 0.0435 |
| Prov-GigaPath alone + Metadata | 0.8979 | 0.9146 | 0.8747 | 0.8957 ± 0.0200 | 0.5361 ± 0.0424 |
| Best Single Model + Metadata | 0.8979 | 0.9146 | 0.8885 | 0.9003 ± 0.0132 | 0.5655 ± 0.0530 |
| Fused (Agent-Selected) (no metadata) | 0.8959 | 0.9105 | 0.8904 | 0.8990 ± 0.0103 | 0.5275 ± 0.0657 |
| **Fused (Agent-Selected) + Metadata** | 0.8979 | 0.9139 | 0.8907 | **0.9008 ± 0.0119** | **0.5487 ± 0.0449** |
| Prism2 VLM alone (no metadata) | 0.8600 | 0.8774 | 0.9191 | 0.8855 ± 0.0304 | 0.4808 ± 0.0796 |
| Prism2 VLM alone + Metadata | 0.8511 | 0.8826 | 0.9136 | 0.8824 ± 0.0312 | 0.5062 ± 0.0728 |
| Quad Multimodal Late Fusion (+ Prism2 VLM) (no metadata) | 0.8921 | 0.9173 | 0.9122 | 0.9072 ± 0.0134 | 0.5705 ± 0.0979 |
| **Quad Multimodal Late Fusion (+ Prism2 VLM) + Metadata (Champion)** | 0.9030 | 0.9184 | 0.9090 | **0.9102 ± 0.0078** | **0.5903 ± 0.0416** |
