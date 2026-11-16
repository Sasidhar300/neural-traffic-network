# External Ingolstadt21 scenario

```sh
python scripts/install_ingolstadt21.py
OMP_NUM_THREADS=1 python -m src.smoke --scenario ingolstadt21
```

The installer downloads the four original scenario files from RESCO v1.0 revision `fc65e91f6711d53296dee824e684b201a3e6808d`. It verifies each file's size and Git blob digest, retains the upstream GPLv3 license text, and installs atomically into ignored `src/networks/ingolstadt21/`. A rerun verifies an existing installation. [Pinned upstream scenario](https://github.com/Pi-Star-Lab/RESCO/tree/fc65e91f6711d53296dee824e684b201a3e6808d/resco_benchmark/environments/ingolstadt21)

| Source checked | Relation to the capstone copy |
|---|---|
| RESCO v1.0 | Identical route, SUMO config and scenario license blobs. Different network blob. |
| RESCO packaged scenario revision `01b969ba` | Same four scenario blobs as v1.0. |
| RESCO v2 revision `f1ed9a17` | Both route and network blobs differ from the capstone copy. |
| InTAS | Original larger Ingolstadt scenario credited by RESCO. No exact capstone network revision established here. |

Both capstone and v1.0 networks contain 21 signal programs. Their signal-ID sets are disjoint. All 163 lane IDs referenced by the capstone's hand-maintained lane map exist upstream, but the old signal keys cannot select those observations on the upstream network. The release discovers the incoming lanes from each upstream signal's controlled links. This can change lane selection, order, state tensor shape, phase behavior and checkpoint compatibility. Train a fresh model on the installed variant. Matching route bytes alone cannot reproduce the original results. No capstone scenario edits or scenario-derived lane mapping are packaged.

On the reference macOS environment, the installer and real CPU/TraCI smoke passed for all 21 signals, including a short PPO update run, saved checkpoint reload and ten evaluation steps per controller. SUMO emitted local projection database warnings but completed all rollouts. This validates runtime compatibility of the release variant. Historical runs remain unverified.
