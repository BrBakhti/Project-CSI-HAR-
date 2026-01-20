# backend/generate_test_csvs.py
import os
import numpy as np
import pandas as pd

output_dir = "uploads/csi"
os.makedirs(output_dir, exist_ok=True)

for i in range(5):
    data = {
        'signal1': np.sin(np.linspace(0, 2 * np.pi, 100)) + np.random.randn(100) * 0.1,
        'signal2': np.cos(np.linspace(0, 2 * np.pi, 100)) + np.random.randn(100) * 0.1,
        'signal3': np.random.rand(100)
    }
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, f"user_sample_{i+1}.csv"), index=False)

print("✅ Fichiers .csv générés dans uploads/csi/")
