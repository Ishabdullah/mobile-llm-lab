# 📱 Mobile LLM Lab

**Train and fine-tune multiple Hugging Face models directly from your phone using Termux.**

This project provides a complete pipeline for fine-tuning Large Language Models (LLMs) using only your mobile device. It integrates GitHub, Hugging Face Hub, and Google Colab to give you a professional ML workflow in your pocket.

## ✨ Features

- 🚀 **One-line training commands** from Termux
- 🔄 **Automatic model management** - creates or updates models on Hugging Face
- 📊 **Multiple model support** - train and manage many models simultaneously
- 💻 **Flexible training options**:
  - Google Colab (free GPU, recommended)
  - Local Termux (CPU only, slower)
  - GitHub Actions (orchestration)
- 📁 **Organized structure** - each model gets its own folder with configs and checkpoints
- 🔐 **Secure token management** via environment variables and GitHub Secrets
- 📈 **Training metrics and logs** saved automatically

## 🏗️ Project Structure

```
mobile-llm-lab/
├── models/                    # Trained models (one folder per model)
│   ├── assistant_v1/
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── training_config.json
│   │   └── training_metrics.json
│   └── classifier_v2/
│       └── ...
├── dataset/                   # Training datasets
│   ├── mydata.txt
│   ├── training.csv
│   └── examples.json
├── train.py                   # Main training script
├── train_colab.ipynb         # Google Colab notebook
├── train_model.sh            # Termux interface script
├── requirements.txt          # Python dependencies
└── .github/
    └── workflows/
        └── train.yml         # GitHub Actions workflow
```

## 🚀 Quick Start

### 1. Prerequisites

**On your phone (Termux):**
```bash
# Install required packages
pkg update && pkg upgrade
pkg install git gh python

# Install Python packages
pip install transformers datasets huggingface_hub torch
```

### 2. Setup Tokens

You need three tokens for full functionality:

#### A. Hugging Face Token
1. Go to https://huggingface.co/settings/tokens
2. Click "New Token"
3. Name: `MobileLLMLab`
4. Permissions: **Write**
5. Copy the token

#### B. GitHub Token
1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Name: `MobileLLMLab`
4. Select scopes: `repo`, `workflow`, `write:packages`
5. Copy the token

#### C. Set Environment Variables (Termux)
```bash
# Add to your ~/.bashrc or ~/.zshrc
echo 'export HF_TOKEN="your_huggingface_token_here"' >> ~/.bashrc
echo 'export HF_USERNAME="your_huggingface_username"' >> ~/.bashrc
echo 'export GH_TOKEN="your_github_token_here"' >> ~/.bashrc

# Reload your shell
source ~/.bashrc
```

### 3. Clone This Repository

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/mobile-llm-lab.git
cd mobile-llm-lab

# Make the script executable
chmod +x train_model.sh
```

### 4. Configure GitHub Secrets

For GitHub Actions and Colab to work, add your tokens as repository secrets:

1. Go to your repo: `https://github.com/YOUR_USERNAME/mobile-llm-lab`
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Add these secrets:
   - `HF_TOKEN`: Your Hugging Face token
   - `HF_USERNAME`: Your Hugging Face username
   - `GH_TOKEN`: Your GitHub token (if not already set)

### 5. Add Training Data

Create a dataset file in the `dataset/` folder:

```bash
# Example: Create a simple text file
cat > dataset/mydata.txt << 'EOF'
This is my training data.
Each line is a training example.
Add as many lines as you need.
EOF

# Commit and push
git add dataset/mydata.txt
git commit -m "Add training dataset"
git push
```

## 📖 Usage

### Simplified Command Format (Recommended)

The easiest way to start training:

```bash
./train_model.sh 'train:model_name="assistant_v1", dataset="dataset/mydata.txt", base_model="distilbert-base-uncased"'
```

**With more options:**
```bash
./train_model.sh 'train:model_name="my_assistant", dataset="dataset/mydata.txt", base_model="gpt2", epochs=5, batch_size=16, task_type="causal_lm"'
```

### Standard Command Format

```bash
./train_model.sh train \
  --model_name "assistant_v1" \
  --base_model "distilbert-base-uncased" \
  --dataset "dataset/mydata.txt" \
  --task_type "causal_lm" \
  --epochs 3 \
  --batch_size 8
```

### Available Commands

```bash
# Start training (triggers GitHub Actions + opens Colab link)
./train_model.sh train --model_name "my_model" --base_model "gpt2" --dataset "dataset/data.txt"

# Train locally on Termux (slow, CPU only)
./train_model.sh local --model_name "my_model" --base_model "gpt2" --dataset "dataset/data.txt"

# Get Google Colab notebook link
./train_model.sh colab

# Show status of models and configuration
./train_model.sh status

# List all models
./train_model.sh list

# Show current configuration
./train_model.sh config

# Show help
./train_model.sh help
```

## 🎯 Training Workflow

When you run a training command, here's what happens:

1. **Script checks** if the model exists on your Hugging Face account
   - If exists: prepares to update it
   - If not: will create a new model repo

2. **Creates model folder** in `models/<model_name>/`

3. **Saves training config** as JSON

4. **Commits and pushes** to GitHub

5. **Triggers GitHub Actions** workflow

6. **Opens Colab notebook link** (if `termux-open-url` available)

7. **You open Colab** and run all cells to train with free GPU

8. **Model is pushed** to Hugging Face Hub automatically

## 🔧 Configuration Options

| Parameter | Description | Default | Required |
|-----------|-------------|---------|----------|
| `model_name` | Name for your fine-tuned model | - | ✅ |
| `base_model` | Base model from Hugging Face Hub | - | ✅ |
| `dataset` | Path to dataset file in repo | - | ✅ |
| `task_type` | Task type: `causal_lm` or `classification` | `causal_lm` | ❌ |
| `epochs` | Number of training epochs | `3` | ❌ |
| `batch_size` | Training batch size | `8` | ❌ |
| `learning_rate` | Learning rate | `2e-5` | ❌ |
| `max_length` | Maximum sequence length | `512` | ❌ |

## 📊 Supported Base Models

You can use any model from Hugging Face Hub. Popular choices:

**Causal Language Models:**
- `gpt2`, `gpt2-medium`, `gpt2-large`
- `distilgpt2`
- `EleutherAI/gpt-neo-125M`
- `EleutherAI/gpt-neo-1.3B`
- `facebook/opt-125m`
- `microsoft/phi-2`

**Classification Models:**
- `distilbert-base-uncased`
- `bert-base-uncased`
- `roberta-base`
- `albert-base-v2`

**Note:** Larger models require more GPU memory. Start with smaller models if training fails.

## 📝 Dataset Formats

The training script supports multiple dataset formats:

### Text Files (`.txt`)
```
Each line is a training example.
Simple and easy to create.
One example per line.
```

### CSV Files (`.csv`)
```csv
text,label
"This is positive",1
"This is negative",0
```

### JSON Files (`.json`)
```json
[
  {"text": "First example", "label": 0},
  {"text": "Second example", "label": 1}
]
```

## 🔐 Security Best Practices

1. **Never commit tokens** to your repository
2. **Use environment variables** in Termux
3. **Use GitHub Secrets** for workflows
4. **Use Colab Secrets** (🔑 icon) in notebooks
5. **Regenerate tokens** if accidentally exposed

## 🐛 Troubleshooting

### "HF_TOKEN not found"
- Make sure you've set the environment variable: `export HF_TOKEN="your_token"`
- Reload your shell: `source ~/.bashrc`

### "Permission denied: ./train_model.sh"
- Make the script executable: `chmod +x train_model.sh`

### "Model training failed in Colab"
- Check GPU is enabled: Runtime → Change runtime type → GPU
- Verify your dataset format is correct
- Check Colab logs for specific error messages
- Try a smaller model or batch size

### "git push rejected"
- Pull latest changes: `git pull`
- Resolve any conflicts
- Push again: `git push`

### "GitHub Actions workflow not found"
- Make sure `.github/workflows/train.yml` exists
- Check the workflow is enabled in GitHub Settings → Actions

## 📚 Examples

### Example 1: Train a Text Generator
```bash
./train_model.sh 'train:model_name="story_generator", dataset="dataset/stories.txt", base_model="gpt2", epochs=5'
```

### Example 2: Train a Classifier
```bash
./train_model.sh 'train:model_name="sentiment_classifier", dataset="dataset/reviews.csv", base_model="distilbert-base-uncased", task_type="classification", epochs=3'
```

### Example 3: Update Existing Model
```bash
# If "my_assistant" already exists on HF, this will update it
./train_model.sh 'train:model_name="my_assistant", dataset="dataset/new_data.txt", base_model="gpt2", epochs=2'
```

### Example 4: Local Training (CPU)
```bash
# For quick testing or very small datasets
./train_model.sh local --model_name "test_model" --base_model "distilgpt2" --dataset "dataset/small.txt" --epochs 1
```

## 🚀 Advanced Usage

### Resume from Checkpoint
Edit the training script to add:
```bash
--resume_from_checkpoint "models/my_model/checkpoint-1000"
```

### Custom Learning Rate
```bash
./train_model.sh 'train:model_name="my_model", dataset="dataset/data.txt", base_model="gpt2", learning_rate=5e-5'
```

### Multiple GPUs (Colab Pro)
Colab Pro provides more powerful GPUs. The training script automatically uses available GPUs.

## 📄 License

MIT License - feel free to use this for any project!

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 💡 Tips

- **Start small**: Begin with small models and datasets to test your pipeline
- **Use Colab**: It's free and much faster than training on your phone
- **Monitor costs**: Be aware of Hugging Face storage limits
- **Save checkpoints**: Configure checkpoint saving to resume interrupted training
- **Track experiments**: Use meaningful model names and keep notes in configs

## 🔗 Resources

- [Hugging Face Documentation](https://huggingface.co/docs)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Google Colab](https://colab.research.google.com/)
- [Termux Documentation](https://wiki.termux.com/)

## 📧 Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Read the Troubleshooting section above

---

**Happy training! 🎉**

Made with ❤️ for mobile ML practitioners
