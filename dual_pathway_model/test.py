"""Run the model once with the parameters in params.yaml."""

import numpy as np
from dual_pathway_model.model import NN, build_and_run, params_base

SEED = 0


def main():
    p = params_base["params"]
    print(
        f"params.yaml  DAYS={p['DAYS']}  TRIALS={p['TRIALS']}  "
        f"N_SYLL={p['N_SYLL']}  ANNEALING={p['ANNEALING']}"
    )
    perf = np.asarray(build_and_run(SEED, params_base, NN), dtype=float)
    print(f"terminal performance (last 100 trials, per syllable): {perf}")
    if not np.isfinite(perf).all():
        raise SystemExit("Run produced non-finite performance")
    print("run ok")


if __name__ == "__main__":
    main()
