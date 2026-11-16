"""Exercise the NTN core on a small generated network."""

from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess

from src.config.data_config import map_configs, signal_configs
from src.config.simulation_config import HyperParameters, SimulationConfig
from src.utils.processing import run_episode
from src.utils.simulation_utils import initialize_agent, initialize_environment

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "synthetic-cross"


def run(config, label, start_time, end_time, *, rl, checkpoint=None):
    env = initialize_environment(config, label, start_time, end_time)
    agent = initialize_agent(config, env, is_rl_agent=rl, model_dir=checkpoint)
    try:
        metrics = run_episode(env, agent)
    finally:
        env.close()
    if not metrics:
        raise RuntimeError(f"No simulator steps for {label}")
    return agent, metrics


def build_synthetic_scenario(workspace):
    subprocess.run(
        ["netconvert", "-n", str(EXAMPLE / "cross.nod.xml"),
         "-e", str(EXAMPLE / "cross.edg.xml"),
         "-o", str(workspace / "cross.net.xml")],
        check=True,
    )
    for name in ("cross.sumocfg", "cross.rou.xml"):
        (workspace / name).write_bytes((EXAMPLE / name).read_bytes())


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=map_configs, default="synthetic-cross")
    args = parser.parse_args()

    with TemporaryDirectory(prefix="ntn-smoke-") as directory:
        workspace = Path(directory)
        settings = map_configs[args.scenario].copy()
        start = settings["start_time"]
        root = SimulationConfig().project_root
        if args.scenario == "synthetic-cross":
            build_synthetic_scenario(workspace)
            settings["net"] = "cross.sumocfg"
            root = workspace
        elif not (root / settings["net"]).is_file():
            parser.error("Install the scenario with python scripts/install_ingolstadt21.py")

        config = SimulationConfig(
            project_root=root,
            map_name=args.scenario,
            training_episodes=1,
            map_config=settings,
            signal_config=signal_configs[args.scenario],
            hyperparameters=HyperParameters(update_interval=4, minibatch_size=4, ppo_epochs=1),
        )
        agent, _ = run(config, "smoke_train", start, start + 80, rl=True)
        checkpoint = workspace / "model"
        agent.save(checkpoint)
        evaluated, rl_metrics = run(config, "smoke_rl", start + 80, start + 180, rl=True, checkpoint=checkpoint)
        _, baseline_metrics = run(config, "smoke_baseline", start + 80, start + 180, rl=False)
        if evaluated.training or len(rl_metrics) != len(baseline_metrics):
            raise RuntimeError("Smoke run did not exercise matched evaluation windows")
        print(
            f"Smoke verified {len(evaluated.obs_act)} signals, checkpoint reload "
            f"and {len(rl_metrics)} evaluation steps per controller."
        )


if __name__ == "__main__":
    main()
