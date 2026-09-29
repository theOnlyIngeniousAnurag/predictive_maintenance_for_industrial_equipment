import pytest
import numpy as np
import os
import json
from src.models.threshold_optimization import compute_threshold_metrics

def test_threshold_calculation():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.4, 0.6])
    threshold = 0.5
    
    metrics = compute_threshold_metrics(y_true, y_prob, threshold)
    # y_pred: [0, 1, 0, 1]
    # TP: 1 (index 3), FP: 1 (index 1), FN: 1 (index 2), TN: 1 (index 0)
    
    assert metrics['tp'] == 1
    assert metrics['fp'] == 1
    assert metrics['fn'] == 1
    assert metrics['tn'] == 1
    assert metrics['recall'] == 0.5
    assert metrics['precision'] == 0.5
