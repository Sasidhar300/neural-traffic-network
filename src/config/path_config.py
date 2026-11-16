from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT.parent / "runs"
PRETRAINED_MODELS_DIR = ROOT.parent / "checkpoints"
