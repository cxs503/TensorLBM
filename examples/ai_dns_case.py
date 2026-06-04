"""DNS-data-driven AI turbulence case: LBM DNS -> train -> AI-LES embed."""
from __future__ import annotations

import json
from pathlib import Path

from tensorlbm import TrainConfig, run_ai_dns_pipeline


def main() -> None:
    res = run_ai_dns_pipeline(
        work_dir=Path("outputs/ai_dns_case"),
        nx=48,
        ny=48,
        tau=0.8,
        c_s=0.1,
        data_steps=60,
        sample_every=10,
        val_steps=40,
        dns_scale=2,
        dns_warmup_steps=20,
        train_config=TrainConfig(epochs=30, batch_size=2048, learning_rate=2e-3),
        seed=0,
        device="cpu",
    )
    print(json.dumps(res.to_dict(), indent=2))


if __name__ == "__main__":
    main()
