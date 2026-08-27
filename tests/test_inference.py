# tests/test_inference.py
"""Unit tests for model inference/prediction logic."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

import pytest
import numpy as np
from main import CLASS_NAMES


def test_class_names_mapping():
    assert CLASS_NAMES[0] == "cat"
    assert CLASS_NAMES[1] == "dog"


def test_prediction_logic_thresholding():
    def get_prediction(prob_dog):
        prob_cat = 1.0 - prob_dog
        prediction = 1 if prob_dog > 0.5 else 0
        confidence = prob_dog if prediction == 1 else prob_cat
        return prediction, confidence

    pred, conf = get_prediction(0.87)
    assert pred == 1  # dog
    assert conf == pytest.approx(0.87)

    pred, conf = get_prediction(0.13)
    assert pred == 0  # cat
    assert conf == pytest.approx(0.87)