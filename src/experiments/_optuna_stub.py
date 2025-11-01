"""Minimal Optuna-compatible stub used when Optuna is unavailable."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


class _BaseSampler:
    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def sample_float(self, low: float, high: float, log: bool) -> float:
        if log:
            low_log = math.log(low)
            high_log = math.log(high)
            value = math.exp(self._rng.uniform(low_log, high_log))
        else:
            value = self._rng.uniform(low, high)
        return value

    def sample_int(self, low: int, high: int) -> int:
        return self._rng.randint(low, high)

    def sample_categorical(self, choices: Sequence[Any]) -> Any:
        return self._rng.choice(list(choices))


class SobolSampler(_BaseSampler):
    pass


class TPESampler(_BaseSampler):
    pass


class QMCSampler(_BaseSampler):
    def __init__(self, *, qmc_type: str = "sobol", seed: Optional[int] = None) -> None:
        super().__init__(seed=seed)
        self.qmc_type = qmc_type


@dataclass
class Trial:
    number: int
    sampler: _BaseSampler
    params: Dict[str, Any] = field(default_factory=dict)

    def suggest_float(self, name: str, low: float, high: float, *, log: bool = False) -> float:
        value = self.sampler.sample_float(low, high, log)
        self.params[name] = value
        return value

    def suggest_int(self, name: str, low: int, high: int) -> int:
        value = self.sampler.sample_int(low, high)
        self.params[name] = value
        return value

    def suggest_categorical(self, name: str, choices: Sequence[Any]) -> Any:
        value = self.sampler.sample_categorical(choices)
        self.params[name] = value
        return value


class Study:
    def __init__(self, direction: str, sampler: _BaseSampler) -> None:
        self.direction = direction
        self.sampler = sampler
        self.trials: List[Trial] = []
        self.best_trial: Optional[Trial] = None
        self.best_value: Optional[float] = None

    def optimize(self, objective, n_trials: int) -> None:
        for number in range(n_trials):
            trial = Trial(number=number, sampler=self.sampler)
            value = objective(trial)
            trial.value = value
            self.trials.append(trial)
            if self.best_value is None:
                self.best_value = value
                self.best_trial = trial
            else:
                is_better = value > self.best_value if self.direction == "maximize" else value < self.best_value
                if is_better:
                    self.best_value = value
                    self.best_trial = trial


class samplers:  # noqa: N801 - mimic optuna namespace
    SobolSampler = SobolSampler
    TPESampler = TPESampler
    QMCSampler = QMCSampler


def create_study(direction: str = "maximize", sampler: Optional[_BaseSampler] = None, **_: Any) -> Study:
    sampler = sampler or SobolSampler()
    return Study(direction=direction, sampler=sampler)

