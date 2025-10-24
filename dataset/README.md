# Dataset Folder

Place your training data files here.

## Supported Formats

- **Text files** (`.txt`): One example per line
- **CSV files** (`.csv`): Must have a `text` column (and optionally `label` for classification)
- **JSON files** (`.json`): Array of objects with `text` field

## Examples

### Text File (example.txt)
```
This is the first training example.
This is the second training example.
Each line becomes one training sample.
```

### CSV File (example.csv)
```csv
text,label
"This is a positive example",1
"This is a negative example",0
```

### JSON File (example.json)
```json
[
  {"text": "First example"},
  {"text": "Second example"},
  {"text": "Third example"}
]
```

## Getting Started

Create your dataset file here, then commit and push:

```bash
# Add your dataset
git add dataset/mydata.txt

# Commit
git commit -m "Add training dataset"

# Push
git push
```

Then use it in training:
```bash
./train_model.sh 'train:model_name="my_model", dataset="dataset/mydata.txt", base_model="gpt2"'
```
