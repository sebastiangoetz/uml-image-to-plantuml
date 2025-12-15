# UML Class Diagram Extractor

## Requirements

- Python 3.9+

## Installation
1. (Optional) Create and activate virtual python environment
    1. ``python -m venv .venv``
    2. Linux/macOS: ``source .venv/bin/activate`` / Windows: ``.venv\Scripts\activate``
2. Run ``pip install -r streamlit/requirements.txt``
### Optional: Enable GPU Acceleration

If you have a CUDA-capable NVIDIA GPU and the correct drivers installed, you can drastically reduce processing time by installing the GPU-enabled PyTorch packages.

First, remove the CPU-only versions (if already installed):
```bash
pip uninstall torch torchvision torchaudio -y
```

Then, install the GPU-enabled versions (example: CUDA 12.1):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For other CUDA versions, see the [official PyTorch installation guide](https://pytorch.org/get-started/locally/). 

## Execution

1. To deploy the streamlit app run:
```bash
cd streamlit
streamlit run app.py
```
2. A browser window for localhost:8501 should open automatically

When you upload an image, the pipeline runs automatically and displays the intermediate results of each step on the page.
If you change the uploaded image or select a different model size, the pipeline is re-executed.
Currently only the model sizes "Nano" and "Medium" are supported, as the "X-Large" Model is too big for the GitLab repository.

| ![](/docs/images/demo.png) |
|:--------------------------:|
| Screenshot of the web app  |
