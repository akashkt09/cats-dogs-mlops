# scripts/post_deployment_check.py
"""
Sends a small batch of labeled test images to the deployed API
and reports real-world accuracy post-deployment.
"""

import requests
import random
import os

API_URL = "http://localhost:8080/predict"
TEST_DIR = "data/test_set/test_set"
SAMPLE_SIZE = 30  # small batch, per assignment's "small batch" requirement

random.seed(42)

results = []

for true_label in ["cats", "dogs"]:
    folder = os.path.join(TEST_DIR, true_label)
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg'))]
    sample = random.sample(files, SAMPLE_SIZE // 2)

    for fname in sample:
        filepath = os.path.join(folder, fname)
        true_class = "cat" if true_label == "cats" else "dog"

        with open(filepath, 'rb') as f:
            response = requests.post(API_URL, files={'file': (fname, f, 'image/jpeg')})

        if response.status_code == 200:
            pred = response.json()['prediction']
            confidence = response.json()['confidence']
            correct = (pred == true_class)
            results.append({
                'file': fname,
                'true_label': true_class,
                'predicted': pred,
                'confidence': confidence,
                'correct': correct
            })
        else:
            print(f"Request failed for {fname}: {response.status_code}")

# Summary
total = len(results)
correct_count = sum(r['correct'] for r in results)
accuracy = correct_count / total if total > 0 else 0

print(f"\nPost-deployment check: {total} requests sent")
print(f"Correct: {correct_count}/{total}")
print(f"Accuracy: {accuracy:.2%}")

# Per-class breakdown
for cls in ['cat', 'dog']:
    cls_results = [r for r in results if r['true_label'] == cls]
    cls_correct = sum(r['correct'] for r in cls_results)
    print(f"  {cls}: {cls_correct}/{len(cls_results)} correct")
