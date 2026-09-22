import numpy as np
from joblib import Parallel, delayed
from lstm_utils import BatchRunModels2Arm

clip_norm = 10.0

model_creator = BatchRunModels2Arm(
    n_train_sessions=30000, n_test_sessions=500, frac_impurity=0.16
)

print("Training structured models...")
Parallel(n_jobs=10)(
    delayed(model_creator.generate_model_data)(
        train_type="structured",
        hidden_size=48,
        lr=0.00002,
        clip_norm=clip_norm,
        name_suffix=i,
    )
    for i in range(10)
)

print("Training unstructured models...")
Parallel(n_jobs=10)(
    delayed(model_creator.generate_model_data)(
        train_type="unstructured",
        hidden_size=48,
        lr=0.00002,
        clip_norm=clip_norm,
        name_suffix=i,
    )
    for i in range(10)
)

print("All models trained and saved.")
