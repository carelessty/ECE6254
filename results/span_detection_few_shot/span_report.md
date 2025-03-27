# Privacy Risk Detection Model Comparison

## Task: Span

### Main Metrics

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| DeepSeek-R1-Distill-Qwen-1.5B (few-shot) | nan | 0.0000 | 0.0000 | 0.0000 |
| RoBERTa-large-self-disclosure (reference) | 0.5000 | 0.0000 | 0.0000 | 0.0000 |

### Detailed Metrics

#### DeepSeek-R1-Distill-Qwen-1.5B (few-shot)

**Confusion Matrix:**

```
Empty confusion matrix
```

**Classification Report:**

```
              precision    recall  f1-score   support


    accuracy                          0.0000   0.0
   macro avg   nan    nan    nan   0.0
weighted avg   nan    nan    nan   0.0
```

#### RoBERTa-large-self-disclosure (reference)

**Confusion Matrix:**

```
[[1, 1],
 [0, 0]]
```

**Classification Report:**

```
              precision    recall  f1-score   support

           0   1.0000    0.5000    0.6667   2.0
           1   0.0000    0.0000    0.0000   0.0

    accuracy                          0.5000   2.0
   macro avg   0.5000    0.2500    0.3333   2.0
weighted avg   1.0000    0.5000    0.6667   2.0
```

