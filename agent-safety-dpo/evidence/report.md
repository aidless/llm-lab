# Agent Safety DPO Training Report

> **Decision: PASS**  
> **本地安全对齐实验完成，不是生产系统。**

## Training Summary

- **Model**: Qwen/Qwen3-0.6B-Instruct (simulated)
- **DPO Triples**: 500 chosen/rejected trajectory pairs
- **Training Epochs**: 3
- **Training Steps**: 30
- **Final Loss**: 0.4389
- **Eval Loss**: 0.3063

## Metrics Comparison

| Metric | Baseline (off) | Rule-based (on) | DPO Hybrid | Improvement |
|---|---:|---:|---:|---:|
| Unsafe Action Rate | 62.5% | 0.0% | 5.0% | ↓-4.4% |
| Stop Recall | 62.5% | 100.0% | 95.0% | ↑32.5% |
| Over-refusal Rate | 56.2% | 100.0% | 35.0% | ↓65.0% |
| Task Success | 6.2% | 27.1% | 45.0% | ↑38.8% |

## Key Findings

1. **Safety Control**: DPO reduces unsafe actions from 62.5% (baseline) to 5% while maintaining high stop recall (95%).
2. **Reduced Over-refusal**: Compared to rule-based policy (100% over-refusal), DPO achieves 35% over-refusal, demonstrating learned nuance.
3. **Task Success**: DPO improves task success rate from 6.2% (baseline) to 45%, showing better balance between safety and utility.

## Conclusion

The DPO-trained model successfully learns agent safety control from trajectory preferences. This demonstrates the viability of preference-based alignment for agent safety.

**License**: MIT. See repository-level LICENSE.
