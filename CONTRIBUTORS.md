# Contributions

Credit for the original capstone belongs to the five-person McMaster University Neural Traffic Networks team (Group 20). The team built the integrated simulation, model and dashboard. Later integration of the final BiLSTM policy and later reward work are separately attributable to teammates in the original history.

The original repository's metadata records the following merged work by **Sasidhar300**. PR and commit identifiers refer to `Boss2002n/capstone_2024`.

| Contribution | Original evidence |
|---|---|
| Early signal-agent, reward and state components | PR #36, `91c6440`, `src/components/{agents,rewards,states}.py`. Earlier action handling in PR #21. |
| Model hyperparameter changes | PR #54, `df309d8`, `src/config/simulation_config.py`. |
| Demand scaling and heavier-traffic evaluation | PRs #53 and #64, `0800ecb` and `1a32205`, original simulation config and main evaluation code. |
| Environment test fixes and plotting/metrics work | PRs #95 and #96, including `98765ff`; plotting PR #114 and metrics PR #135, local squash `bc32cc4`. |
| Design, verification and README document submissions | PRs #27, #76, and #153. Credit for the documents remains with the team where jointly authored. |

Release preparation, the reward-configuration fix, test repairs and the synthetic example are follow-on work prepared with AI assistance. They are outside the original capstone contribution record.
