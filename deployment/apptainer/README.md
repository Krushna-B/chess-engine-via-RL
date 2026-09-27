# Apptainer

Build the image from the repository root on a node with Apptainer:

```bash
apptainer build --fakeroot chess-engine.sif \
  deployment/apptainer/chess-engine.def
```

The image provides Python 3.12, `uv`, CMake, and native build tools. The host
NVIDIA driver and CUDA libraries are exposed at runtime with `--nv`.

The Slurm jobs use the image only when `APPTAINER_IMAGE` is set. Repository and
artifact paths must be visible at the same absolute paths inside and outside
the container.
