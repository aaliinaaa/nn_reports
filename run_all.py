import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import yaml

STUDENT_ID = "8ВМ52400"
PROFILE = f"configs/profiles/{STUDENT_ID}.yaml"
CONFIG = "configs/lab0.yaml"

Path("reports").mkdir(exist_ok=True)
log_file = Path("reports/run_all.log")

def log(msg: str):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

def run_cmd(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        log(f"ERROR in command: {' '.join(cmd)}")
        log(f"STDERR:\n{proc.stderr}")
        sys.exit(1)
    lines = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
    return lines[-1]

def run_train(label: str, extra_args: list[str] | None = None) -> str:
    log(f"== train: {label} ==")
    cmd = [sys.executable, "-m", "src.train", "--config", CONFIG, "--profile", PROFILE]
    if extra_args:
        cmd.extend(extra_args)
    last_line = run_cmd(cmd)
    data = json.loads(last_line)
    run_id = data["run_id"]
    with open(f"reports/runs_{label}.txt", "a", encoding="utf-8") as f:
        f.write(run_id + "\n")
    log(f"  -> run_id={run_id}")
    return run_id

def main():
    log_file.write_text("", encoding="utf-8")
    prof = yaml.safe_load(Path(PROFILE).read_text(encoding="utf-8"))
    cfg = yaml.safe_load(Path(CONFIG).read_text(encoding="utf-8"))

    base_seed = int(prof["profile"]["base_seed"])
    K = int(cfg["train"]["epochs"])
    half = K // 2
    log(f"base_seed={base_seed} epochs={K} half={half}")

    # == E1: repeatability ==
    Path("reports/runs_e1.txt").write_text("", encoding="utf-8")
    run_train("e1")
    run_train("e1")

    # == E2: variance across 5 seeds ==
    Path("reports/runs_e2.txt").write_text("", encoding="utf-8")
    for k in range(5):
        run_train("e2", ["--override", f"experiment.seed={base_seed + k}"])

    # == E3: checkpoint/resume ==
    Path("reports/runs_e3a.txt").write_text("", encoding="utf-8")
    Path("reports/runs_e3b.txt").write_text("", encoding="utf-8")
    run_train("e3a", ["--override", f"train.epochs={K}"])
    e3b_first = run_train("e3b", ["--override", f"train.epochs={half}"])

    log(f"== resume e3b to {K} epochs ==")
    last_line = run_cmd([
        sys.executable, "-m", "src.train",
        "--config", CONFIG, "--profile", PROFILE,
        "--resume", e3b_first, "--override", f"train.epochs={K}"
    ])
    e3b_resumed = json.loads(last_line)["run_id"]
    with open("reports/runs_e3b.txt", "a", encoding="utf-8") as f:
        f.write(e3b_resumed + "\n")
    log(f"  -> resumed run_id={e3b_resumed}")

    # == CF2: num_workers=2 ==
    Path("reports/runs_cf2.txt").write_text("", encoding="utf-8")
    run_train("cf2", ["--override", "data.num_workers=2"])
    run_train("cf2", ["--override", "data.num_workers=2"])

    # == CF3: truncated checkpoint resume ==
    Path("reports/runs_cf3.txt").write_text("", encoding="utf-8")
    ckpt_path = f"runs/{e3b_first}/checkpoints/last.pt"
    trunc_path = f"runs/{e3b_first}/checkpoints/model_only.pt"
    log(f"== truncate checkpoint: {trunc_path} ==")
    from src.engine.checkpoint import truncate_to_model_only
    truncate_to_model_only(ckpt_path, trunc_path)

    last_line = run_cmd([
        sys.executable, "-m", "src.train",
        "--config", CONFIG, "--profile", PROFILE,
        "--resume", e3b_first, "--truncate-checkpoint", trunc_path,
        "--override", f"train.epochs={K}"
    ])
    cf3_run = json.loads(last_line)["run_id"]
    with open("reports/runs_cf3.txt", "a", encoding="utf-8") as f:
        f.write(cf3_run + "\n")
    log(f"  -> truncated-resume run_id={cf3_run}")

    log("==[done] all 12 runs complete==")

if __name__ == "__main__":
    main()
