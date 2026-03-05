# detection-pipeline
Generic detection pipeline

## Installation

Create a virtual environment and install runtime dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the pipeline with a source (image, video, stream URL, or camera index):

```bash
python run.py --source example_media/drones --show
```

Use `--show` to enable visualization.
