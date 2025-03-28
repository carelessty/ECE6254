# Privacy Risk Detection Model Comparison

## Task: Span

### Main Metrics

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| DeepSeek-R1-Distill-Qwen-1.5B (fine-tuning) | nan | 0.0000 | 0.0000 | 0.0000 |
| RoBERTa-large-self-disclosure (reference) | 0.5000 | 0.0000 | 0.0000 | 0.0000 |

### Detailed Metrics

#### DeepSeek-R1-Distill-Qwen-1.5B (fine-tuning)

**Confusion Matrix:**

```
[0, 0]
[0, 0]
```

**Classification Report:**

```
     Label  Precision     Recall   F1-Score    Support


  Accuracy     0.0000
 macro avg        nan        nan        nan        0.0
weighted avg        nan        nan        nan        0.0
```

#### RoBERTa-large-self-disclosure (reference)

**Confusion Matrix:**

```
[1, 1]
[0, 0]
```

**Classification Report:**

```
     Label  Precision     Recall   F1-Score    Support

         0     1.0000     0.5000     0.6667        2.0
         1     0.0000     0.0000     0.0000        0.0

  Accuracy     0.5000
 macro avg     0.5000     0.2500     0.3333        2.0
weighted avg     1.0000     0.5000     0.6667        2.0
```

