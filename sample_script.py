from datasets import load_dataset

# Load the dataset
dataset = load_dataset("Gen-Verse/ReasonFlux-F1-SFT")

# Save each split to a separate JSON file
for split in dataset:
    dataset[split].to_json(f"ReasonFlux_{split}.json", orient="records", lines=True)
