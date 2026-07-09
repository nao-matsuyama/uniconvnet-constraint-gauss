# UniConvNet Versioned Workspace

This workspace is a clean scaffold for versioning the UniConvNet project using Git, Git LFS, and Docker.

Structure
- src/: project source code
- data/: dataset pointers (do not store large data here unless tracked via Git LFS)
- models/: trained model artifacts (use Git LFS)
- experiments/: experiment outputs and logs
- docker/: Dockerfiles and helper scripts

Usage
1. Initialize git: `git init`
2. Configure Git LFS for *.pth files: `git lfs track "*.pth"`
3. Build Docker image: `docker build -t uniconvnet:latest .`

Train the U-Net with the provided backbone checkpoint:

```bash
cd src
python train.py \
  --data-dir /workspace/scinti_segmentation \
  --pretrained ../uniconvnet_t_1k_224_ema.pth
```

You can also pass `--freeze-backbone` if you want to keep the pretrained encoder fixed.
