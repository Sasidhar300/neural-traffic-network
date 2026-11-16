# Neural Traffic Networks

Neural Traffic Networks (NTN) is a McMaster University Group 20 capstone on multi-agent reinforcement learning for traffic signals. Each intersection observes nearby lanes and chooses a green phase with a PFRL PPO policy backed by a PyTorch BiLSTM. The team compared the learned controller with a round-robin fixed-interval baseline in Eclipse SUMO.

[Historical results and limitations](RESULTS.md) · [Contributions and team credit](CONTRIBUTORS.md) · [Third-party notices](THIRD_PARTY_NOTICES.md) · [Install Ingolstadt21](SCENARIO.md)

## How it works

`SimulationConfig` supplies network, seed, PPO, and reward settings. The simulator executes yellow and green phases, collects per-signal observations, and supplies the resulting transitions to independent PPO agents during training. The metrics modules aggregate waiting, queue, speed, reward, and emissions-related diagnostics. The historical dashboard visualized those summaries. The release includes the Django dashboard and an authored synthetic junction. The [external installation](SCENARIO.md) adds a pinned RESCO network without bundling third-party scenario files.

## Run the release

Use Python 3.12 and SUMO with `sumo` and `netconvert` on PATH. The checked reference environment uses SUMO 1.21.0 and the direct Python dependencies listed in `requirements.txt`.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q src/test
OMP_NUM_THREADS=1 python -m src.smoke
```

The smoke run starts a local TraCI connection. It builds a temporary four-way network, performs a short CPU-compatible PPO run, saves and reloads a checkpoint, then runs both controllers for the same ten-step evaluation window. The printed result describes functionality. The [historical results](RESULTS.md) describe the separate Ingolstadt21 comparisons.

## Local Django dashboard

```sh
python src/website/manage.py migrate
python src/website/manage.py test dashboard.test.test_views --noinput
python src/website/manage.py runserver 127.0.0.1:8000
```

Register a local account using the app, then choose a scenario and train a model. SQLite and run outputs are ignored. Local settings generate a process-specific signing key unless `NTN_DJANGO_SECRET_KEY` is provided in the process environment. The app uses one background worker per server process. This demo setup is for loopback use. The release uses TraCI. Windows, Linux, the SUMO GUI and libsumo were not validated in this release.

The release fixes dropped reward parameters and duplicate active-signal entries on reset. Historical baseline phase handling, reward normalization and emissions units remain limitations. Checkpoints are saved by scenario. The original network, checkpoints and administrative records are outside this curated release.

## Verification scope

Reference environment: macOS arm64, Python 3.12.4, Django 5.2, SUMO 1.21.0 and CPU PyTorch 2.6.0. The commands above passed 19 core tests and 17 Django view tests. The external scenario smoke completed for all 21 signals. A two-episode synthetic run of the dashboard backend also exported the metrics JSON and all 21 comparison plots. The Django view tests use a test database and mock the background worker. Browser interaction and the original benchmark replay remain open.

## Licensing

The original team work is all rights reserved. See the [copyright notice](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
