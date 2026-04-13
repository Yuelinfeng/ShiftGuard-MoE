from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(slots=True)
class AccessResult:
    hit: bool
    evicted: int | None = None


class ExpertCachePolicy(ABC):
    def __init__(self, capacity: int, name: str) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.name = name
        self.steps = 0

    @abstractmethod
    def access(self, expert_id: int) -> AccessResult:
        raise NotImplementedError

    @abstractmethod
    def snapshot(self) -> list[int]:
        raise NotImplementedError


class LRUExpertCache(ExpertCachePolicy):
    def __init__(self, capacity: int) -> None:
        super().__init__(capacity=capacity, name="lru")
        self._cache: OrderedDict[int, None] = OrderedDict()

    def access(self, expert_id: int) -> AccessResult:
        self.steps += 1
        if expert_id in self._cache:
            self._cache.move_to_end(expert_id)
            return AccessResult(hit=True)

        evicted = None
        if len(self._cache) >= self.capacity:
            evicted, _ = self._cache.popitem(last=False)
        self._cache[expert_id] = None
        return AccessResult(hit=False, evicted=evicted)

    def snapshot(self) -> list[int]:
        return list(self._cache.keys())


class LFUExpertCache(ExpertCachePolicy):
    def __init__(self, capacity: int) -> None:
        super().__init__(capacity=capacity, name="lfu")
        self._cache: set[int] = set()
        self._frequency: dict[int, int] = {}
        self._last_used: dict[int, int] = {}

    def access(self, expert_id: int) -> AccessResult:
        self.steps += 1

        if expert_id in self._cache:
            self._frequency[expert_id] += 1
            self._last_used[expert_id] = self.steps
            return AccessResult(hit=True)

        evicted = None
        if len(self._cache) >= self.capacity:
            evicted = min(
                self._cache,
                key=lambda item: (self._frequency[item], self._last_used[item]),
            )
            self._cache.remove(evicted)
            self._frequency.pop(evicted, None)
            self._last_used.pop(evicted, None)

        self._cache.add(expert_id)
        self._frequency[expert_id] = 1
        self._last_used[expert_id] = self.steps
        return AccessResult(hit=False, evicted=evicted)

    def snapshot(self) -> list[int]:
        return sorted(self._cache)


@dataclass(slots=True)
class PromptCacheStats:
    hits: int
    misses: int
    by_layer: dict[int, dict[str, float | int]]

    @property
    def accesses(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        if self.accesses == 0:
            return 0.0
        return self.hits / self.accesses


class LayeredCacheSimulator:
    def __init__(self, policy_cls: type[ExpertCachePolicy], num_layers: int, capacity: int) -> None:
        self.policy_name = policy_cls(capacity).name
        self.layers = {layer_idx: policy_cls(capacity) for layer_idx in range(num_layers)}
        self.total_hits = 0
        self.total_misses = 0

    def consume(self, layer_accesses: dict[str, list[int]]) -> PromptCacheStats:
        total_hits = 0
        total_misses = 0
        by_layer: dict[int, dict[str, float | int]] = {}

        for layer_key, expert_ids in sorted(layer_accesses.items(), key=lambda item: int(item[0])):
            layer_idx = int(layer_key)
            policy = self.layers[layer_idx]
            layer_hits = 0
            layer_misses = 0
            for expert_id in expert_ids:
                result = policy.access(expert_id)
                if result.hit:
                    layer_hits += 1
                else:
                    layer_misses += 1

            layer_accesses_count = layer_hits + layer_misses
            by_layer[layer_idx] = {
                "hits": layer_hits,
                "misses": layer_misses,
                "hit_rate": 0.0 if layer_accesses_count == 0 else layer_hits / layer_accesses_count,
            }
            total_hits += layer_hits
            total_misses += layer_misses

        self.total_hits += total_hits
        self.total_misses += total_misses
        return PromptCacheStats(hits=total_hits, misses=total_misses, by_layer=by_layer)

    @property
    def cumulative_hit_rate(self) -> float:
        accesses = self.total_hits + self.total_misses
        if accesses == 0:
            return 0.0
        return self.total_hits / accesses
