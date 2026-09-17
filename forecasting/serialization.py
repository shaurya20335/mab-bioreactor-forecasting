"""Read both original unit-labelled results and generic-target result files."""

import json
from pathlib import Path


TARGET_UNITS = {
    "volume_ml": "mL", "viable_cells_1e6_ml": "10^6 cells/mL", "mab_mg_L": "mg/L",
    "glucose_g_L": "g/L", "lactate_g_L": "g/L", "viability_pct": "%",
}


def load_mab_results(path):
    """Normalize metric names for the mAb-specific notebook/report only.

    Old artifacts have unit-bearing keys; generic training results carry
    units in metadata. Do not relabel a glucose or other target as mAb.
    """
    data = json.loads(Path(path).read_text())
    if data.get("target", "mab_mg_L") != "mab_mg_L":
        raise ValueError("This report is specific to mAb. Other targets require their own units and report.")
    aliases = {"rmse": "rmse_mg_L", "mae": "mae_mg_L", "mse": "mse_mg2_L2",
               "bias": "bias_mg_L", "radius": "radius_mg_L", "mean_batch_rmse": "mean_batch_rmse_mg_L"}

    def normalize(value):
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {key: normalize(item) for key, item in value.items()}
        for generic, labelled in aliases.items():
            if generic in result:
                if labelled in result and result[labelled] != result[generic]:
                    raise ValueError(f"Conflicting metric values for {generic} and {labelled}.")
                result[labelled] = result[generic]
        return result

    return normalize(data)
