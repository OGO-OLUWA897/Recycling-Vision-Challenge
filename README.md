# Recycling-Vision-Challenge
Smart Recycling Vision System using the TrashNet dataset to classify recyclable waste (cardboard, glass, metal, paper, plastic, and trash) with baseline and improved AI models, including robustness testing on noisy images.

================================================================================
===========================================

DESCRIPTION
-----------
Two-task waste image classifier using the TrashNet dataset (6 classes).

  Task 1 – Baseline:  Random Forest classifier on flattened pixel features
  Task 2 – Improved:  SVM with RBF kernel on HOG features


REQUIRED LIBRARIES
------------------
Install all dependencies with:

    pip install -r requirements.txt

Requires Python 3.9, 3.10, or 3.11.


DATASET LAYOUT
--------------
Place the provided dataset folder in the project root:

    dataset/
    ├── train/
    │   ├── cardboard/
    │   ├── glass/
    │   ├── metal/
    │   ├── paper/
    │   ├── plastic/
    │   └── trash/
    ├── val/          (same structure)
    ├── test/         (same structure)
    └── noisy-test/   (same structure)


HOW TO RUN
----------

Step 1 – Install dependencies:
    pip install -r requirements.txt

Step 2 – (Optional) Explore the dataset:
    python explore_dataset.py
    Outputs: outputs/exploration/

Step 3 – Train and evaluate the baseline (Task 1):
    python task1_baseline.py
    Outputs: outputs/task1/

Step 4 – Train and evaluate the improved model (Task 2):
    python task2_improved.py
    Outputs: outputs/task2/

Step 5 – Re-evaluate any saved model:
    python evaluate.py --model outputs/task1/baseline_model.pkl \
                       --task  1 \
                       --split noisy-test \
                       --out   outputs/eval


EXPECTED OUTPUTS
----------------
Task 1 (outputs/task1/):
  baseline_model.pkl               Saved Random Forest model
  confusion_matrix_validation.png
  confusion_matrix_test.png
  confusion_matrix_noisy_test.png
  report_validation.txt            Classification report (P/R/F1 per class)
  report_test.txt
  report_noisy_test.txt
  misclassified_noisy_test.png     Grid of misclassified examples

Task 2 (outputs/task2/):
  improved_model.pkl               Saved SVM model
  feature_scaler.pkl               Saved StandardScaler
  confusion_matrix_*.png
  report_*.txt
  misclassified_noisy_test.png
  comparison_baseline_vs_improved.png


FILE DESCRIPTIONS
-----------------
  task1_baseline.py     Random Forest on flattened 64×64 RGB pixels
  task2_improved.py     SVM+RBF on HOG features with StandardScaler
  evaluate.py           CLI tool to re-run evaluation on any saved model
  explore_dataset.py    Dataset visualisation and class distribution plots
  requirements.txt      Python dependencies
  README.txt            This file


TECHNICAL DETAILS
-----------------
Task 1 – Baseline:
  Image size:   64×64 pixels
  Features:     Flattened RGB pixels → 12,288 dimensions
  Model:        RandomForestClassifier(n_estimators=100, max_depth=20)
  Training:     ~2 minutes on CPU

Task 2 – Improved:
  Image size:   64×64 pixels
  Features:     HOG descriptors → 1,764 dimensions
                (9 orientations, 8×8 pixels/cell, 2×2 cells/block)
  Scaling:      StandardScaler (zero mean, unit variance)
  Model:        SVC(kernel='rbf', C=10, gamma='scale')
  Training:     ~5–10 minutes on CPU

No GPU required. All computation runs on CPU via scikit-learn.
