# COCO-CMFD

A scalable, fully local dataset generation pipeline for creating high-quality Copy-Move Forgery Detection (CMFD) benchmark datasets. This utilizes MS COCO 2017 to produce tri-map ground truth forgery imagery designed for training modern dual-branch neural networks like BusterNet.

## Project Layout

```
coco-cmfd/
├── README.md
├── LICENSE             - MIT License
├── generator/          - Single-process, WSL-safe generation logic
│   ├── generator.py    - Core augmentation engine
│   ├── filter_annotations.py - Subprocess utility to extract annotation subset safely
│   ├── visualize.py    - Matplotlib helper to plot samples headlessly
│   └── main.py         - Entry point
├── docs/               - Academic methodologies and procedure definitions
├── examples/           - Visualizations and sample datasets
├── metadata_schema.json
└── output/             - Flushed imagery and labels
    ├── tampered/
    ├── binary_masks/
    ├── logs/
    └── metadata.csv
```

## Running the Pipeline

Before running, ensure `pycocotools`, `opencv-python`, `pandas` and `matplotlib` are installed. The pipeline is designed to be WSL memory safe on 8GB RAM environments.

Run from the root directory:

```bash
python generator/main.py
```
