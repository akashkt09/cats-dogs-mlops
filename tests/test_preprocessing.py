# tests/test_preprocessing.py
"""Unit tests for image preprocessing logic."""

import io
import numpy as np
import pytest
from PIL import Image

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from main import preprocess_image, IMG_SIZE


def make_fake_image_bytes(size=(300, 200), color=(255, 0, 0)):
    """Create an in-memory fake image for testing, no real file needed."""
    img = Image.new('RGB', size, color)
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


def test_preprocess_image_resizes_correctly():
    image_bytes = make_fake_image_bytes(size=(500, 300))
    result = preprocess_image(image_bytes)
    assert result.shape == (1, IMG_SIZE[0], IMG_SIZE[1], 3)


def test_preprocess_image_normalizes_pixel_values():
    image_bytes = make_fake_image_bytes()
    result = preprocess_image(image_bytes)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_preprocess_image_handles_grayscale_input():
    # a grayscale image should still be converted to 3-channel RGB
    img = Image.new('L', (100, 100), 128)  # 'L' = grayscale mode
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    result = preprocess_image(buf.getvalue())
    assert result.shape == (1, IMG_SIZE[0], IMG_SIZE[1], 3)