import os
import pickle
from pathlib import Path
from typing import Any, Optional

import pandas as pd

try:  # using wandb where available
  import wandb
except ImportError:  # pragma: no cover
  wandb = None

def _load_dotenv(dotenv_path: Path | str | None = None) -> None:
  path = Path(dotenv_path) if dotenv_path else Path(__file__).resolve().parent.parent / ".env"
  if not path.is_file():
    return
  for raw in path.read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
      continue
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip().strip('"').strip("'")
    os.environ.setdefault(key, value)


_load_dotenv()


class _ExperimentLogger:
  def __init__(self):
    self.log_directory: Optional[Path] = None
    self.prefix: Optional[str] = None
    self.run_dir: Optional[Path] = None
    self._run = None
    self._mode = os.getenv("WANDB_MODE", "offline")
    self._project = os.getenv("WANDB_PROJECT", "conditional-density-estimation")

  def configure(self, *args, log_directory: Optional[str] = None, prefix: Optional[str] = None,
                project: Optional[str] = None, mode: Optional[str] = None, **kwargs) -> None:
    if args:
      if log_directory is None:
        log_directory = args[0]
      if len(args) > 1 and prefix is None:
        prefix = args[1]

    assert log_directory, "log_directory must be provided"
    assert prefix, "prefix must be provided"

    self.log_directory = Path(log_directory).resolve()
    self.prefix = prefix
    self.run_dir = self.log_directory / self.prefix
    self.run_dir.mkdir(parents=True, exist_ok=True)

    project = project or self._project
    mode = mode or os.getenv("WANDB_MODE", self._mode)
    os.environ.setdefault("WANDB_MODE", mode)
    os.environ.setdefault("WANDB_PROJECT", project)

    if wandb is not None:
      if self._run:
        wandb.finish()
      try:
        self._run = wandb.init(project=project, name=prefix, dir=str(self.run_dir), mode=mode,
                               return_previous=True, **kwargs)
      except TypeError as exc:
        if "return_previous" in str(exc) or "finish_previous" in str(exc):
          self._run = wandb.init(project=project, name=prefix, dir=str(self.run_dir), mode=mode,
                                 reinit=True, **kwargs)
        else:
          raise
    else:
      self._run = None

  def _resolve_path(self, path: str) -> Path:
    target = Path(path)
    if not target.is_absolute():
      assert self.run_dir is not None
      target = self.run_dir / target
    target.parent.mkdir(parents=True, exist_ok=True)
    return target

  def log(self, *values: Any, **kwargs: Any) -> None:
    print(*values, **kwargs)

  def log_line(self, *lines: Any) -> None:
    print(*lines)

  def dump_pkl(self, data: Any, path: str) -> None:
    target = self._resolve_path(path)
    with open(target, "wb") as f:
      pickle.dump(data, f)
    if self._run and wandb is not None:
      wandb.save(str(target), base_path=str(self.run_dir))

  def load_pkl(self, path: str) -> Any:
    target = self._resolve_path(path)
    with open(target, "rb") as f:
      return pickle.load(f)

  def load_pkl_log(self, path: str) -> Any:
    return self.load_pkl(path)

  def save_dataframe(self, df: pd.DataFrame, file_name: str = "results.csv") -> None:
    target = self._resolve_path(file_name)
    header = not target.exists()
    df.to_csv(target, mode="a", header=header, index=False)

  def log_pyplot(self, key: str, fig: Any) -> None:
    target = self._resolve_path(f"{key}.png")
    fig.savefig(target)
    if self._run and wandb is not None:
      wandb.log({key: wandb.Image(str(target))})

  def flush(self, file_name: Optional[str] = None) -> None:
    pass

  def log_pkl(self, data: Any, path: str) -> None:
    self.dump_pkl(data, path)

logger = _ExperimentLogger()

