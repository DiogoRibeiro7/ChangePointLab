# Online vs. Offline Detection

This note compares real-time detection with batch segmentation.

## Methods
- **BOCPD**: online Bayesian inference
- **PELT/E-Divisive**: offline global optimization/nonparametric segmentation

## Trade-offs
| Aspect | BOCPD (Online) | PELT/E-Divisive (Offline) |
|--------|----------------|---------------------------|
| Latency | Immediate | After full data seen |
| Accuracy | Depends on hazard | Optimal (given model) |
| Memory | Constant | Grows with data |
| Complexity | O(T) | PELT O(T), E-Divisive O(T^2) |

## Streaming Considerations
- Update hazards to reflect expected changepoint frequency
- Use pruning or windowing to limit memory
- Evaluate detection delay vs. false alarm rate

## Hybrid Approaches
Combine coarse offline segmentation with BOCPD for fine-grained online refinement.

```mermaid
graph LR;
    A[Raw Stream] --> B[PELT Pre-Segmentation];
    B --> C[BOCPD Refinement];
    C --> D[Alerts];
```
