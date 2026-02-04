# BERT Error Analysis

## Summary

**Model**: BERT-base-uncased fine-tuned on QEvasion
**Test Accuracy**: 65.26%
**Test F1 (Macro)**: 0.56
**Total Errors**: 107 / 308 (34.7%)

---

## Error Breakdown by Class

| True Class | Total | Errors | Error Rate | Primary Confusion |
|------------|-------|--------|------------|-------------------|
| Clear Reply | 79 | 43 | 54.4% | Misclassified as Ambivalent (86%) |
| Clear Non-Reply | 23 | 8 | 34.8% | Misclassified as Ambivalent (87.5%) |
| Ambivalent | 206 | 56 | 27.2% | Misclassified as Clear Reply (62.5%) |

Clear Reply has the highest error rate at 54.4%, with the vast majority confused as Ambivalent.

## Confusion Matrix

```
                  Predicted
True             Amb    CNR     CR
Ambivalent       150     21     35    (73% recall)
Clear Non-Reply    7     15      1    (65% recall)
Clear Reply       37      6     36    (46% recall)
```

72 errors (67% of all errors) involve confusion between Ambivalent and Clear Reply. Only 27 involve Clear Non-Reply. The model is better at detecting non-replies than distinguishing degrees of clarity.

## Confidence Analysis

| Prediction Type | Mean Confidence | Median Confidence |
|----------------|-----------------|-------------------|
| Correct | 0.7315 | 0.7273 |
| Errors | 0.7387 | 0.7441 |

67 high-confidence errors (>0.7 confidence) account for 62.6% of all errors. The model is frequently confident and wrong, suggesting poor calibration. Temperature scaling or other calibration techniques could help.

## Text Length Effect

| Prediction Type | Mean Answer Length | Median Answer Length |
|----------------|-------------------|---------------------|
| Correct | 2035 chars | 1795 chars |
| Errors | 1520 chars | 1463 chars |

Errors occur disproportionately on shorter answers (25% shorter on average), likely because shorter answers provide less context for the model.

## Linguistic Patterns in Errors

| Pattern | In Correct | In Errors | Difference |
|---------|-----------|----------|------------|
| Uncertainty markers | 7.0% | 11.2% | +4.2% |
| Hedging language | 38.8% | 40.2% | +1.4% |
| Yes-but constructions | 39.8% | 37.4% | -2.4% |
| Deflection phrases | 6.5% | 1.9% | -4.6% |
| Question dodging | 10.0% | 5.6% | -4.3% |

Uncertainty phrases ("don't know", "not sure") appear more frequently in misclassified samples, as these markers can indicate either Ambivalent or Clear Non-Reply depending on context.

## Implications for Model Improvement

1. **Ambiguity in labels**: The Ambivalent/Clear Reply boundary is genuinely difficult. Politicians intentionally give ambivalent answers that sound clear.
2. **Class imbalance**: Training is 59% Ambivalent, biasing the model toward this class as a safe default.
3. **Overconfidence**: High confidence on errors suggests the model needs calibration (label smoothing, temperature scaling).
4. **Short answers**: Less context leads to more errors; data augmentation or attention to answer length could help.

## What BERT Does Well

- Clear Non-Reply detection (65% recall, only 8 errors) -- obvious evasions are well-captured
- Ambivalent detection (73% recall, 0.75 F1) -- majority class with strong performance
- Captures semantic evasion patterns beyond keyword matching
