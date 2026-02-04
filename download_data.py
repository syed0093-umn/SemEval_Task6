from datasets import load_dataset
import os

print("Downloading QEvasion dataset from HuggingFace...")
print("This will download the actual data files (not Git LFS pointers)")
print()

# Download to a local directory
dataset = load_dataset("ailsntua/QEvasion", cache_dir="./cache")

# Save locally
dataset.save_to_disk("./QEvasion")

print(f"\n* Dataset downloaded to: {os.path.abspath('./QEvasion')}")
print(f"\n* Training examples: {len(dataset['train'])}")
print(f"* Test examples: {len(dataset['test'])}")

# Clean up cache if desired
# import shutil
# shutil.rmtree("./cache")
# print("\n* Cache cleaned up")
