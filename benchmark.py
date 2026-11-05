"""Bounded kernel cost profile; not a performance experiment or result claim."""
from pathlib import Path
import json
import time
import numpy as np
from gridshift.environment import model
from gridshift.experiment import get_map, suite_configs
from gridshift.engine import train, evaluate


def main():
    spec = get_map("open25")
    nxt, rewards, terminal = model(spec)
    uniforms = np.random.default_rng(991).random((5000, 2))
    eval_uniforms = np.random.default_rng(992).random((200, 250, 2))
    empty_evaluation = np.empty((0, 1, 250, 2))
    phi = np.zeros(625)
    def training_only():
        return train(np.zeros((625, 4)), nxt, rewards, terminal, phi, 0, 250,
                     .1, .95, .05, .05, 5001, False, uniforms, empty_evaluation)
    warm_start = time.perf_counter()
    training_only()
    evaluate(np.zeros((625, 4)), nxt, rewards, terminal, 0, 250, .95, .05, eval_uniforms)
    warmup = time.perf_counter()-warm_start
    start = time.perf_counter()
    for _ in range(100): training_only()
    training_seconds = time.perf_counter()-start
    start = time.perf_counter()
    evaluation_steps = 0
    for _ in range(20):
        obs = evaluate(np.zeros((625, 4)), nxt, rewards, terminal, 0, 250, .95, .05, eval_uniforms)
        evaluation_steps += int(obs[:, 2].sum())
    evaluation_seconds = time.perf_counter()-start
    train_rate = 500000/training_seconds
    eval_rate = evaluation_steps/evaluation_seconds
    suites = {}
    for name in ("reconstruction", "core"):
        configs = suite_configs(name)
        training_steps = sum(c.steps for c in configs)
        maximum_eval_steps = sum(c.steps//c.eval_every*c.eval_episodes*250 for c in configs)
        suites[name] = {"runs": len(configs), "training_steps": training_steps,
                        "max_evaluation_steps": maximum_eval_steps,
                        "kernel_seconds_estimate_at_measured_rates": training_steps/train_rate+maximum_eval_steps/eval_rate}
    report = {"purpose": "cost profile only; no learned-performance inference",
              "warmup_or_cache_load_seconds": warmup,
              "training_only_steps": 500000, "training_only_seconds": training_seconds,
              "training_steps_per_second": train_rate,
              "evaluation_only_steps": evaluation_steps,
              "evaluation_only_seconds": evaluation_seconds,
              "evaluation_steps_per_second": eval_rate,
              "estimates_exclude": "result IO, RNG generation, preparation, import/startup and system contention",
              "suite_upper_step_counts": suites}
    target = Path("verification/pilot_cost_profile.json")
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
