Developer notes

- Use `uv` to create and manage the virtual environment. Example:

```
uv create .venv
uv activate .venv
uv install -r requirements.txt
```

If `uv` is not present, `python3 -m venv .venv` and `source .venv/bin/activate` work.

- To run training in Docker with GPU access (NVIDIA runtime):

```
docker run --gpus all --ipc=host -v $(pwd):/workspace -it uniconvnet:local /bin/bash
```

- To tag images with git hash:

```
GITHASH=$(git rev-parse --short HEAD)
docker build -t uniconvnet:$GITHASH .
```
