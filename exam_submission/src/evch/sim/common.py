from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(slots=True)
class SimulationResult:
    metrics: pd.DataFrame
    summary: dict[str, Any]
