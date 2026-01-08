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

def load_campaigns(yaml_path: str):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    defaults = cfg.get("defaults", {})
    raw_campaigns = cfg["campaigns"]
    params = cfg["params"]

    campaigns = []
    for idx, c in enumerate(raw_campaigns):
        campaign = defaults.copy()
        campaign.update(c)
        campaign["params"] = params

        # Assign a stable internal index
        campaign["_index"] = idx

        # Ensure required keys exist
        missing = DEFAULT_KEYS - campaign.keys()
        if missing:
            raise ValueError(
                f"Campaign {idx} missing keys: {missing}"
            )

        campaigns.append(campaign)

    return campaigns


def campaign_to_study_name(campaign: dict) -> str:
    # Prefer explicit name if provided
    if "name" in campaign:
        return f"hpo__{campaign['name']}"

    # Fallback: deterministic name
    parts = [
        f"{k}={campaign[k]}"
        for k in sorted(campaign)
        if not k.startswith("_")
    ]
    return "hpo__" + "__".join(parts)
