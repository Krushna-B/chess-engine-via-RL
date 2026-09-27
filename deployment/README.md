# Deployment

Remote execution definitions live here. The current Modal application builds a
headless CUDA image, mounts persistent storage at `artifacts/`, and runs the
existing sequential generation loop.

Run commands from the repository root:

```bash
modal run --detach deployment/modal_app.py
modal run deployment/modal_app.py::list-checkpoints
modal run deployment/modal_app.py::download --model models/gen_0001_jit.pt
```

The deployment layer should orchestrate engine and training components; model
architecture and chess logic should remain in `training/` and `src/`.

