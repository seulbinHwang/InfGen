"""Create small synthetic fixtures; these are NOT WOMD or scientific evaluations."""
from argparse import ArgumentParser
from pathlib import Path
import pickle
import torch


def make_scenario(index=0):
    agents, frames, lanes, points = 8, 91, 3, 121
    time = torch.arange(frames, dtype=torch.float32) / 10
    position = torch.zeros(agents, frames, 3)
    position[:, :, 0] = 2 * time[None] + torch.arange(agents)[:, None] * 6 - 12
    position[:, :, 1] = (torch.arange(agents) % lanes)[:, None] * 3
    valid = torch.ones(agents, frames, dtype=torch.bool)
    valid[6, :25] = False  # A late arrival exercises spatial supervision.
    valid[7, 65:] = False  # An exit exercises temporal control supervision.
    role = torch.zeros(agents, 3, dtype=torch.bool)
    role[0, 0] = True
    role[:, 2] = True
    x = torch.linspace(-30, 90, points)
    map_position = torch.zeros(lanes, points, 3)
    map_position[:, :, 0] = x
    map_position[:, :, 1] = torch.arange(lanes)[:, None] * 3
    return {
        "scenario_id": f"synthetic_{index:04d}",
        "agent": {
            "num_nodes": agents, "av_idx": 0, "id": torch.arange(agents),
            "role": role, "valid_mask": valid,
            "type": torch.zeros(agents, dtype=torch.uint8),
            "category": torch.ones(agents, dtype=torch.uint8),
            "position": position, "heading": torch.zeros(agents, frames),
            "velocity": torch.tensor([2., 0.]).expand(agents, frames, 2).clone(),
            "shape": torch.tensor([4.3, 1.8, 1.]).expand(agents, frames, 3).clone(),
        },
        "map_polygon": {"num_nodes": lanes, "type": torch.zeros(lanes, dtype=torch.uint8),
                        "light_type": torch.zeros(lanes, dtype=torch.uint8)},
        "map_point": {"num_nodes": lanes * points, "position": map_position.reshape(-1, 3),
                      "orientation": torch.zeros(lanes * points),
                      "magnitude": torch.ones(lanes * points), "height": torch.zeros(lanes * points),
                      "type": torch.zeros(lanes * points, dtype=torch.uint8)},
        ("map_point", "to", "map_polygon"): {
            "edge_index": torch.stack([torch.arange(lanes * points),
                                       torch.arange(lanes).repeat_interleave(points)])},
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--count", type=int, default=2)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    for split in ("training", "validation"):
        folder = args.output / split
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(args.count):
            destination = folder / f"synthetic_{index:04d}.pkl"
            with destination.open("xb") as stream:
                pickle.dump(make_scenario(index), stream)
    print(f"Synthetic fixtures saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
