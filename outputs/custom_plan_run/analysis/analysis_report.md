# Analysis Report

## Experiment

- Mode: `real`
- Model: `deepseek-ai/DeepSeek-V2-Lite-Chat`
- Cache capacity: `8`
- Number of prompts: `24`
- Number of MoE layers traced: `26`

## Stable Baseline

- `LRU` stable total hit rate: `0.3613`
- `LFU` stable total hit rate: `0.2474`

## Scenario Comparison

| Scenario | Policy | Total Hit Rate | Post-shift Hit Rate | Drop vs Stable | Worst Window |
| --- | --- | ---: | ---: | ---: | ---: |
| stable | LRU | 0.3613 | 0.3609 | 0.0000 | 0.3608 |
| stable | LFU | 0.2474 | 0.2476 | 0.0000 | 0.2447 |
| lru_break | LRU | 0.3558 | 0.3525 | 0.0055 | 0.3490 |
| lru_break | LFU | 0.2262 | 0.2219 | 0.0212 | 0.2132 |
| lfu_break | LRU | 0.3562 | 0.3503 | 0.0051 | 0.3490 |
| lfu_break | LFU | 0.2282 | 0.2161 | 0.0192 | 0.2132 |
| mixed_shift | LRU | 0.3555 | 0.3519 | 0.0058 | 0.3464 |
| mixed_shift | LFU | 0.2612 | 0.2571 | -0.0138 | 0.2506 |

## Takeaways

- Largest degradation: `lru_break` with `LFU` showing `drop_vs_stable=0.0212`.
- Most resilient non-stable case: `mixed_shift` with `LFU` at `drop_vs_stable=-0.0138`.
- `lru_break`: `LRU drop=0.0055`, `LFU drop=0.0212`.
- `lfu_break`: `LRU drop=0.0051`, `LFU drop=0.0192`.
- `mixed_shift`: `LRU drop=0.0058`, `LFU drop=-0.0138`.
