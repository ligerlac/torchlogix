import yaml

DEFAULT_KEYS = {
    "dataset",
    "architecture",
    "batch_size",
    "num_iterations",
    "eval_freq",
    "weight_init",
    "connections",
    "lut_rank",
}

def load_studies(yaml_path: str):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    defaults = cfg.get("defaults", {})
    raw_studies = cfg["studies"]
    params = cfg["params"]
    name = cfg["name"]

    studies = []
    for idx, c in enumerate(raw_studies):
        study = defaults.copy()
        study.update(c)
        study["params"] = params
        study["campaign_name"] = name

        # Assign a stable internal index
        study["_index"] = idx

        # Ensure required keys exist
        missing = DEFAULT_KEYS - study.keys()
        if missing:
            raise ValueError(
                f"Study {idx} missing keys: {missing}"
            )

        studies.append(study)

    return studies


def study_to_campaign_name(study: dict) -> str:
    # Prefer explicit name if provided
    if "name" in study:
        return f"{study['campaign_name']}__{study['name']}"

    # Fallback: deterministic name
    parts = [
        f"{k}={study[k]}"
        for k in sorted(study)
        if not k.startswith("_")
    ]
    return f"{study['campaign_name']}__" + "__".join(parts)
