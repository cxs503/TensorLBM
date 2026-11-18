"""Generate a matched free-slip diagnostic case from the upstream showcase."""

import argparse
import math
import re
from pathlib import Path

from benchmarks.pending.dam_break_mm_channel.run import (
    DENSITY,
    END_T,
    GRAVITY,
    NU,
    SURFACE_TENSION,
    WIDTH_M,
)


def make_case(template: str, n: int, celerity: float) -> str:
    tau = 0.5 + 3 * NU * celerity * n / (WIDTH_M * math.sqrt(2 * GRAVITY * WIDTH_M))
    template = re.sub(r"(<)\s*50,\s*50,\s*1\s*(>)", f"< {n}, {n}, 1 >", template, count=1)
    parameters = {
        "columnWidth": n,
        "galileiNumber": GRAVITY * WIDTH_M**3 / NU**2,
        "bondNumber": DENSITY * GRAVITY * WIDTH_M**2 / SURFACE_TENSION,
        "relaxationRate": 1 / tau,
        "timesteps": math.ceil(END_T * n / celerity) + 1,
        "evaluationFrequency": 1,
        "smagorinskyConstant": 0,
        "writeFrequency": 0,
        "loadBalancingFrequency": 0,
        "printLoadBalancingStatistics": "false",
    }
    for name, value in parameters.items():
        template, count = re.subn(
            rf"^([ \t]*{name}\s+)[^;]+;",
            lambda match: f"{match[1]}{value};",
            template,
            flags=re.MULTILINE,
        )
        if not count:
            raise ValueError(f"missing parameter {name} in pinned upstream configuration")
    return template


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--celerity", type=float, default=0.06)
    args = parser.parse_args()
    if args.cells < 4 or not 0 < args.celerity < 0.15:
        parser.error("invalid resolution or lattice celerity")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(make_case(args.template.read_text(), args.cells, args.celerity))


if __name__ == "__main__":
    main()
