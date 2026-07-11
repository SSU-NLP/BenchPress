"""Main experiment (Part 2, 본실험) entrypoint.

Runs the v3 alignment loop on `data/labels_part2`, reusing `I_star.txt` +
`cognitive_abilities.json` produced by the pre-experiment
(`autotagging_loop/pretrain.py`, Part 1). When `experiment.enable_v_loop=True`,
every iteration regenerates V via the Executer; otherwise V stays fixed at the
pre-experiment seed.
"""

from __future__ import annotations

from autotagging_loop.runner.cli import main


if __name__ == "__main__":
    main()
