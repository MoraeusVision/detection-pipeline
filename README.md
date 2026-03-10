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

Run the pipeline with a project config:

```bash
python run.py --config projects/example_project/config.json
```

The project config keeps project-specific values together, such as source,
detector type, and model path. Paths inside the config are resolved relative to
the config file, which makes it easier to keep each project's config and model
files in the same folder.
