import os
# FORCE CPU ONLY TO PREVENT MUTEX CRASH
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_USE_LEGACY_KERAS"] = "0"

import keras
import sys

print("Checking environment...")
try:
    model = keras.models.load_model('my_model.keras', compile=False)
    print("\n--- SUCCESS! ---")
    model.summary()
except Exception as e:
    print(f"Failed to load: {e}")