"""Configuration for the original synthetic example in this release."""

map_configs = {
    "synthetic-cross": {
        "net": "examples/synthetic-cross/cross.sumocfg",
        "step_length": 10,
        "yellow_length": 3,
        "step_ratio": 1,
        "start_time": 0,
        "end_time": 180,
    },
    "ingolstadt21": {
        "net": "networks/ingolstadt21/ingolstadt21.sumocfg",
        "step_length": 10,
        "yellow_length": 3,
        "step_ratio": 1,
        "start_time": 57600,
        "end_time": 61200,
    }
}
signal_configs = {
    "synthetic-cross": {
        "1": {
            "lane_sets": {"N-S": ["north_in_0"], "S-N": ["south_in_0"],
                          "E-W": ["east_in_0"], "W-E": ["west_in_0"]},
            "downstream": {},
        }
    }
}

# Use the installed SUMO program to discover upstream lane IDs.
signal_configs["ingolstadt21"] = {}
