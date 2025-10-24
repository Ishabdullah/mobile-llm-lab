# Mobile LLM Lab - Setup Instructions

## ✅ What Has Been Done

Your complete ML training pipeline is ready:
- ✅ GitHub repository created: https://github.com/Ishabdullah/mobile-llm-lab
- ✅ All code files written and pushed
- ✅ Project structure organized
- ✅ Documentation complete

## 🔧 Manual Steps Required

### Step 1: Create Hugging Face Token (Required)

1. Visit: https://huggingface.co/settings/tokens
2. Click "New Token"
3. Name: `MobileLLMLab`
4. Permission: **Write**
5. Copy the token

Then in Termux:
```bash
echo 'export HF_TOKEN="hf_xxxxxxxxxxxxx"' >> ~/.bashrc
echo 'export HF_USERNAME="Ishabdullah"' >> ~/.bashrc
source ~/.bashrc
```

### Step 2: Add GitHub Secrets (Required for Colab)

1. Visit: https://github.com/Ishabdullah/mobile-llm-lab/settings/secrets/actions
2. Click "New repository secret"
3. Add these secrets:
   - Name: `HF_TOKEN`, Value: your Hugging Face token
   - Name: `HF_USERNAME`, Value: `Ishabdullah`

### Step 3: Update Colab Notebook (First Time Only)

1. Open: https://colab.research.google.com/github/Ishabdullah/mobile-llm-lab/blob/main/train_colab.ipynb
2. In the first code cell, change:
   ```python
   GITHUB_REPO = "Ishabdullah/mobile-llm-lab"  # This is correct!
   ```
3. Add secrets in Colab:
   - Click the key icon (🔑) on the left sidebar
   - Add `HF_TOKEN` with your Hugging Face token
   - Add `HF_USERNAME` with `Ishabdullah`
   - Add `GH_TOKEN` if you want to push results back to GitHub

## 🚀 Test Your Setup

### Quick Test Command
```bash
cd ~/mobile-llm-lab
./train_model.sh 'train:model_name="test_model", dataset="dataset/sample_data.txt", base_model="distilgpt2"'
```

This will:
1. Create model configuration
2. Push to GitHub
3. Give you a Colab link
4. Open the link in your browser (if termux-api installed)

### In Google Colab:
1. Make sure GPU is enabled (Runtime → Change runtime type → GPU)
2. Add your secrets (🔑 icon)
3. Run all cells (Runtime → Run all)
4. Wait for training to complete
5. Your model will appear at: https://huggingface.co/Ishabdullah/test_model

## 📋 Command Cheat Sheet

```bash
# Check status
./train_model.sh status

# Train a model
./train_model.sh 'train:model_name="my_model", dataset="dataset/data.txt", base_model="gpt2"'

# List all models
./train_model.sh list

# Get Colab link
./train_model.sh colab

# Show help
./train_model.sh help
```

## 🔗 Important Links

- **Repository:** https://github.com/Ishabdullah/mobile-llm-lab
- **Colab Notebook:** https://colab.research.google.com/github/Ishabdullah/mobile-llm-lab/blob/main/train_colab.ipynb
- **GitHub Secrets:** https://github.com/Ishabdullah/mobile-llm-lab/settings/secrets/actions
- **Hugging Face Tokens:** https://huggingface.co/settings/tokens
- **Your HF Profile:** https://huggingface.co/Ishabdullah

## ❓ Need Help?

- Read the full README: https://github.com/Ishabdullah/mobile-llm-lab/blob/main/README.md
- Check the troubleshooting section
- Review example commands

## 🎉 You're Ready!

Once you complete the 3 manual steps above, you'll be able to train models with a single command from Termux!
