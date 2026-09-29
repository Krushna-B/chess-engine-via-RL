import os


class WandbLogger:
    def __init__(self, enabled: bool, config: dict):
        self._run = None
        if not enabled:
            return

        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("WANDB_ENABLED=1 requires the wandb package") from error

        init_kwargs = {
            "project": os.environ.get("WANDB_PROJECT", "chess-engine"),
            "name": os.environ.get("RUN_ID"),
            "config": config,
        }
        entity = os.environ.get("WANDB_ENTITY")
        mode = os.environ.get("WANDB_MODE")
        if entity:
            init_kwargs["entity"] = entity
        if mode:
            init_kwargs["mode"] = mode
        run_id = os.environ.get("WANDB_RUN_ID")
        group = os.environ.get("WANDB_GROUP")
        tags = os.environ.get("WANDB_TAGS")
        if run_id:
            init_kwargs["id"] = run_id
            init_kwargs["resume"] = os.environ.get("WANDB_RESUME", "allow")
        if group:
            init_kwargs["group"] = group
        if tags:
            init_kwargs["tags"] = [tag.strip() for tag in tags.split(",") if tag.strip()]
        self._run = wandb.init(**init_kwargs)

    def log(self, metrics: dict, step: int | None = None) -> None:
        if self._run is not None:
            self._run.log(metrics, step=step)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
