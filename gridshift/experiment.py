"""Configuration-led runs. Every saved observation comes from an executed run."""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import hashlib
import json
import platform
import time
import sys
import numpy as np
from .environment import GridSpec, model, distances
from .potentials import exponential, corrupt, smooth, shift, describe
from .engine import train, evaluate

ROOT = Path(__file__).resolve().parents[1]
ENGINE_FILES = ("environment.py", "learner.py", "potentials.py", "engine.py", "experiment.py")


def fingerprint():
    h = hashlib.sha256()
    for name in ENGINE_FILES:
        h.update(name.encode())
        h.update((Path(__file__).parent / name).read_bytes())
    return h.hexdigest()


def seed_for(seed, role, map_name):
    return int.from_bytes(hashlib.sha256(f"{seed}:{role}:{map_name}".encode()).digest()[:8], "little")


def get_map(name, reward="goal"):
    if name == "open25":
        return GridSpec(name=name, reward=reward)
    if name == "open10":
        return GridSpec(size=10, name=name, reward=reward)
    if name == "detour10":
        walls = tuple([r * 10 + 3 for r in range(9)] +
                      [r * 10 + 6 for r in range(1, 10)])
        return GridSpec(size=10, walls=walls, name=name, reward=reward)
    raise ValueError(f"unknown map: {name}")


@dataclass(frozen=True)
class RunConfig:
    phase: str = "pilot"
    map_name: str = "open25"
    reward: str = "goal"
    method: str = "unshaped"
    normalisation: str = "unit"
    corruption: float = 0.0
    seed: int = 0
    q_initial: float = 0.0
    gamma: float = 0.95
    alpha: float = 0.1
    epsilon: float = 0.05
    evaluation_epsilon: float = 0.05
    steps: int = 5000
    eval_every: int = 250
    eval_episodes: int = 10
    cutoff: str = "bootstrap"
    smoothing_strength: float = 1.0

    def __post_init__(self):
        if self.method not in ("unshaped", "pbrs", "shifted", "smooth_pbrs", "smooth_shifted"):
            raise ValueError("unknown method")
        if self.normalisation not in ("unit", "literal"):
            raise ValueError("unknown normalisation")
        if self.cutoff not in ("bootstrap", "episodic_cutoff"):
            raise ValueError("unknown cutoff")
        if self.steps <= 0 or self.eval_every <= 0 or self.steps % self.eval_every:
            raise ValueError("steps must be a positive multiple of eval_every")
        if self.eval_episodes < 1 or not 0 < self.alpha <= 1 or not 0 <= self.gamma < 1:
            raise ValueError("invalid numeric settings")
        if not (0 <= self.epsilon <= 1 and 0 <= self.evaluation_epsilon <= 1):
            raise ValueError("invalid exploration")
        if not (0 <= self.corruption <= 1 and 0 <= self.smoothing_strength <= 1):
            raise ValueError("invalid corruption/smoothing")
        if not np.isfinite(self.q_initial):
            raise ValueError("nonfinite initialisation")


def canonical(config):
    """Remove factors that cannot change an unshaped run; do not pseudoreplicate."""
    if config.method == "unshaped":
        return replace(config, normalisation="unit", corruption=0.0)
    return config


def run_id(config):
    encoded = json.dumps(asdict(canonical(config)), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def prepare_potential(config, spec):
    reference = exponential(spec, config.normalisation)
    corrupted = corrupt(reference, spec, config.corruption,
                        np.random.default_rng(seed_for(config.seed, "corruption", spec.name)))
    before = describe(corrupted, reference, spec)
    base = (smooth(corrupted, spec, config.smoothing_strength)
            if config.method.startswith("smooth_") else corrupted.copy())
    after = describe(base, reference, spec)
    if config.method == "unshaped":
        potential, b = np.zeros_like(base), 0.0
    elif config.method in ("shifted", "smooth_shifted"):
        potential, b = shift(base, spec, config.q_initial, config.gamma,
                             0.0 if spec.reward == "goal" else -1.0)
    else:
        potential, b = base.copy(), 0.0
    return potential, {"before": before, "after": after, "b": b,
                       "shifted_sha256": hashlib.sha256(potential.tobytes()).hexdigest()}


def execute(config, output=None, resume=True):
    config = canonical(config)
    rid = run_id(config)
    source_hash = fingerprint()
    directory = Path(output) if output else None
    if directory:
        directory.mkdir(parents=True, exist_ok=True)
        metadata_path = directory / f"{rid}.json"
        if resume and metadata_path.exists():
            old = json.loads(metadata_path.read_text())
            if old["engine_sha256"] != source_hash:
                raise RuntimeError("existing result belongs to different source; preserve and use a new output directory")
            return old
    started = time.perf_counter()
    spec = get_map(config.map_name, config.reward)
    if not np.isfinite(distances(spec)[spec.start]):
        raise ValueError("unsolvable start/goal")
    nxt, rewards, terminal = model(spec)
    phi, potential_record = prepare_potential(config, spec)
    q = np.full((spec.size**2, 4), config.q_initial, dtype=np.float64)
    q[terminal] = 0.0
    q[list(spec.walls)] = 0.0
    train_rng = np.random.default_rng(seed_for(config.seed, "training", spec.name))
    eval_rng = np.random.default_rng(seed_for(config.seed, "evaluation", spec.name))
    training_uniforms = train_rng.random((config.steps, 2))
    checkpoints = config.steps // config.eval_every
    evaluation_uniforms = eval_rng.random((checkpoints, config.eval_episodes, spec.max_steps, 2))
    prepared = time.perf_counter()
    observations, q, goals, cutoffs = train(
        q, nxt, rewards, terminal, phi, spec.start, spec.max_steps,
        config.alpha, config.gamma, config.epsilon, config.evaluation_epsilon,
        config.eval_every, config.cutoff == "episodic_cutoff", training_uniforms, evaluation_uniforms)
    computed = time.perf_counter()
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(observations)):
        raise FloatingPointError("nonfinite result")
    evaluation_steps = int(observations[:, :, 2].sum())
    meta = {"run_id": rid, "config": asdict(config), "engine_sha256": source_hash,
            "potential": potential_record,
            "metrics": {"mean_checkpoint_return": float(observations[:, :, 0].mean()),
                        "final_return": float(observations[-1, :, 0].mean()),
                        "final_success": float(observations[-1, :, 3].mean()),
                        "mean_success": float(observations[:, :, 3].mean()),
                        "evaluation_steps": evaluation_steps,
                        "training_steps": config.steps,
                        "checkpoints": checkpoints,
                        "evaluation_episodes": checkpoints * config.eval_episodes,
                        "training_goals": int(goals), "training_cutoffs": int(cutoffs)},
            "timing": {"preparation_seconds": prepared-started,
                       "kernel_seconds_including_evaluation": computed-prepared},
            "runtime": {"python": sys.version, "numpy": np.__version__,
                        "platform": platform.platform(), "processor": platform.processor()}}
    if directory:
        np.savez_compressed(directory / f"{rid}.npz", observations=observations, q_final=q,
                            potential=phi, checkpoints=np.arange(1, checkpoints+1)*config.eval_every)
        meta["timing"]["total_seconds_including_save"] = time.perf_counter()-started
        metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    else:
        meta["timing"]["total_seconds_including_save"] = time.perf_counter()-started
    return meta


def suite_configs(suite):
    if suite == "pilot":
        return [RunConfig(seed=9001, method="unshaped"),
                RunConfig(seed=9001, method="shifted", normalisation="literal", q_initial=1),
                RunConfig(seed=9001, method="smooth_shifted", corruption=.75,
                          map_name="detour10", q_initial=-1, eval_every=1000,
                          evaluation_epsilon=0)]
    configs = []
    if suite == "reconstruction":
        for reward in ("goal", "step"):
            for cutoff in ("bootstrap", "episodic_cutoff"):
                for normalisation in ("literal", "unit"):
                    for initial in (-1.0, 0.0, 1.0):
                        for seed in range(5):
                            for method in ("unshaped", "pbrs", "shifted"):
                                configs.append(RunConfig(phase=suite, reward=reward, cutoff=cutoff,
                                    normalisation=normalisation, q_initial=initial, seed=seed,
                                    method=method, steps=250000))
    elif suite == "core":
        for map_name in ("open10", "detour10"):
            for normalisation in ("literal", "unit"):
                for fraction in (0.0, .25, .75):
                    for initial in (-1.0, 0.0, 1.0):
                        for seed in range(10):
                            for method in ("unshaped", "pbrs", "shifted", "smooth_pbrs", "smooth_shifted"):
                                configs.append(RunConfig(phase=suite, map_name=map_name,
                                    normalisation=normalisation, corruption=fraction,
                                    q_initial=initial, seed=seed, method=method,
                                    steps=50000, eval_every=1000, evaluation_epsilon=0))
    else:
        raise ValueError("unknown suite")
    unique = {run_id(canonical(c)): canonical(c) for c in configs}
    return list(unique.values())


def random_baselines(output):
    records = []
    for map_name in ("open25", "open10", "detour10"):
        for reward in ("goal", "step"):
            spec = get_map(map_name, reward)
            nxt, rewards, terminal = model(spec)
            q = np.zeros((spec.size**2, 4))
            for seed in range(10):
                u = np.random.default_rng(seed_for(seed, "random_baseline", map_name)).random((100, spec.max_steps, 2))
                obs = evaluate(q, nxt, rewards, terminal, spec.start, spec.max_steps, .95, 1.0, u)
                records.append({"map_name": map_name, "reward": reward, "seed": seed,
                                "episodes": 100, "return": float(obs[:, 0].mean()),
                                "success": float(obs[:, 3].mean())})
                np.savez_compressed(Path(output)/f"random_{map_name}_{reward}_{seed}.npz", observations=obs)
    (Path(output)/"random_baselines.json").write_text(json.dumps(records, indent=2))
    return records
