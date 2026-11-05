"""Export simulated dataset and an explicitly informed symbolic reference."""
from dataclasses import asdict
from pathlib import Path
from collections import deque
import json
import numpy as np
from gridshift.environment import model, distances
from gridshift.experiment import get_map
from gridshift.engine import evaluate


def bfs_path(spec):
    nxt, _, _ = model(spec)
    queue = deque([spec.start]); previous = {spec.start: None}; expanded = 0
    while queue:
        state = queue.popleft(); expanded += 1
        if state == spec.goal:
            path = []
            while state is not None:
                path.append(state); state = previous[state]
            return path[::-1], expanded
        for ns in nxt[state]:
            ns = int(ns)
            if ns not in previous:
                previous[ns] = state; queue.append(ns)
    raise ValueError("goal unreachable")


def main():
    directory = Path("data"); directory.mkdir(exist_ok=True)
    records=[]; baselines=[]
    for name in ("open25", "open10", "detour10"):
        for reward in ("goal", "step"):
            spec = get_map(name, reward); nxt, rewards, terminal = model(spec)
            path, expanded = bfs_path(spec)
            dist = distances(spec)
            q = -dist[nxt]
            q[~np.isfinite(q)] = -1e9
            obs = evaluate(q, nxt, rewards, terminal, spec.start, 250, .95, 0.,
                           np.random.default_rng(7).random((100, 250, 2)))
            stem=f"{name}_{reward}"
            np.savez_compressed(directory/f"{stem}.npz", next_states=nxt, original_rewards=rewards,
                                true_terminal=terminal, bfs_distances=dist)
            np.savez_compressed(directory/f"bfs_{stem}.npz", observations=obs)
            records.append({**asdict(spec), "file":f"{stem}.npz", "state_encoding":"row*size+col",
                            "actions":["up","right","down","left"],
                            "collection_cutoff_is_true_terminal":False})
            baselines.append({"map_name":name,"reward":reward,"path":path,
                              "shortest_steps":len(path)-1,"expanded_states":expanded,
                              "mean_return":float(obs[:,0].mean()),"success":float(obs[:,3].mean()),
                              "information":"full exact transition graph; informed reference, no training"})
    (directory/"environments.json").write_text(json.dumps(records,indent=2))
    (directory/"symbolic_baselines.json").write_text(json.dumps(baselines,indent=2))
    print(json.dumps(baselines,indent=2))


if __name__ == "__main__": main()
