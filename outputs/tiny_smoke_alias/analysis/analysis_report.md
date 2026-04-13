# Analysis Report

## Experiment

- Mode: `tiny-random`
- Model: `tiny-random-deepseek-v2`
- Cache capacity: `4`
- Number of prompts: `8`
- Number of MoE layers traced: `4`

## Stable Baseline

- `LRU` stable total hit rate: `0.4954`
- `LFU` stable total hit rate: `0.5255`

## Scenario Comparison

| Scenario | Policy | Total Hit Rate | Post-shift Hit Rate | Drop vs Stable | Worst Window |
| --- | --- | ---: | ---: | ---: | ---: |
| stable | LRU | 0.4954 | 0.5099 | 0.0000 | 0.4879 |
| stable | LFU | 0.5255 | 0.5857 | 0.0000 | 0.4985 |
| block_shift | LRU | 0.4618 | 0.3958 | 0.0336 | 0.3958 |
| block_shift | LFU | 0.5486 | 0.5083 | -0.0231 | 0.5083 |
| medium_shift | LRU | 0.4618 | 0.3958 | 0.0336 | 0.3958 |
| medium_shift | LFU | 0.5486 | 0.5083 | -0.0231 | 0.5083 |
| severe_shift | LRU | 0.4255 | 0.4000 | 0.0698 | 0.4000 |
| severe_shift | LFU | 0.4628 | 0.5000 | 0.0627 | 0.4485 |

## Takeaways

- Largest degradation: `severe_shift` with `LRU` showing `drop_vs_stable=0.0698`.
- Most resilient non-stable case: `block_shift` with `LFU` at `drop_vs_stable=-0.0231`.
- `block_shift`: `LRU drop=0.0336`, `LFU drop=-0.0231`.
- `medium_shift`: `LRU drop=0.0336`, `LFU drop=-0.0231`.
- `severe_shift`: `LRU drop=0.0698`, `LFU drop=0.0627`.
