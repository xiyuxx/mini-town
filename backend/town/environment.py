"""What each layer of the mind may see of the world around an agent.

One record is kept per agent — the observation the engine refreshes on arrival,
on change, and on a clock — and every reader is a projection of it. Declaring
the projections here, rather than describing the surroundings again in each
layer, keeps "who can see what" in one reviewable place: they used to drift, and
a layer that described the room itself ended up knowing only the people in it.
"""

ALL = "*"

# Readers are prompt builders. Rule checks read the record directly, because they
# ask about a state of the record rather than rendering it for a model.
ENVIRONMENT_VIEWS: dict[str, dict[str, object]] = {
    "planning": {
        "location": ALL,
        "target_id": ALL,
        "entities": ALL,
        "nearby_agents": ALL,
        "available_processes": ALL,
        "weather": ALL,
        "observed_at": ALL,
    },
    "dialogue": {
        "location": ALL,
        # A person does not read out the id of the machine beside them, and does
        # not recite its capabilities either. The id stays on the people only,
        # to match the speakers against the record.
        "entities": ("name", "kind"),
        "nearby_agents": ("id", "status", "action"),
        "weather": ALL,
        "observed_at": ALL,
    },
}


def view(record: dict | None, reader: str) -> dict:
    """Project one observation record for one reader.

    Unknown readers raise: a new layer declares what it sees rather than quietly
    receiving everything.
    """
    if reader not in ENVIRONMENT_VIEWS:
        raise KeyError(f"unknown environment reader: {reader}")
    if not isinstance(record, dict):
        return {}
    projected: dict = {}
    for name, wanted in ENVIRONMENT_VIEWS[reader].items():
        if name not in record:
            continue
        value = record[name]
        if wanted is ALL:
            projected[name] = value
        elif isinstance(value, list):
            if not all(isinstance(item, dict) for item in value):
                projected[name] = value
                continue
            projected[name] = [
                {key: item[key] for key in wanted if key in item} for item in value
            ]
        elif isinstance(value, dict):
            projected[name] = {key: value[key] for key in wanted if key in value}
        else:
            projected[name] = value
    return projected
