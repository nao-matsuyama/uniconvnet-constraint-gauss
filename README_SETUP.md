Quick setup

1. Initialize git and Git LFS

```bash
cd /autohome/bonescinti/nao/uniconvnet_versioned
git init
git lfs install
git lfs track "*.pth"
```

2. Create virtualenv via `uv` (uv is a thin wrapper for venv you mentioned)

```bash
uv create .venv  # or: python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`SimpleITK` is required for dataset loading and validation.

3. Build Docker image

```bash
cd docker
docker build -t uniconvnet:local .
```

4. Usage notes
- Put large pre-trained models into `models/` and commit via Git LFS.
- Keep data outside repo or in `data/` but excluded by .gitignore; document download steps.
