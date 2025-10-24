#!/data/data/com.termux/files/usr/bin/bash
#
# Mobile LLM Lab - Termux Interface Script
# Usage: ./train_model.sh [command] [options]
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$SCRIPT_DIR"

# Function to print colored messages
print_header() {
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Function to check if HF model exists
check_hf_model_exists() {
    local model_name="$1"
    local hf_username="$2"

    if [ -z "$hf_username" ]; then
        print_warning "HF_USERNAME not set, skipping model existence check"
        return 1
    fi

    local repo_id="${hf_username}/${model_name}"

    print_info "Checking if model exists: $repo_id"

    # Use curl to check if model repo exists
    local response=$(curl -s -o /dev/null -w "%{http_code}" \
        "https://huggingface.co/api/models/${repo_id}" \
        -H "Authorization: Bearer ${HF_TOKEN}")

    if [ "$response" = "200" ]; then
        print_success "Model exists: $repo_id (will update)"
        return 0
    else
        print_info "Model does not exist: $repo_id (will create new)"
        return 1
    fi
}

# Function to parse simplified command format
parse_simple_command() {
    local cmd="$1"

    # Remove "train:" prefix if present
    cmd="${cmd#train:}"

    # Parse key-value pairs
    IFS=',' read -ra PARAMS <<< "$cmd"

    for param in "${PARAMS[@]}"; do
        # Trim whitespace
        param=$(echo "$param" | xargs)

        # Extract key and value
        key=$(echo "$param" | cut -d'=' -f1 | xargs)
        value=$(echo "$param" | cut -d'=' -f2- | xargs)

        # Remove quotes from value
        value="${value%\"}"
        value="${value#\"}"
        value="${value%\'}"
        value="${value#\'}"

        # Set variables
        case "$key" in
            model_name) MODEL_NAME="$value" ;;
            dataset) DATASET="$value" ;;
            base_model) BASE_MODEL="$value" ;;
            task_type) TASK_TYPE="$value" ;;
            epochs) EPOCHS="$value" ;;
            batch_size) BATCH_SIZE="$value" ;;
            learning_rate) LEARNING_RATE="$value" ;;
            *) print_warning "Unknown parameter: $key" ;;
        esac
    done
}

# Function to create model directory
create_model_dir() {
    local model_name="$1"
    local model_dir="${REPO_DIR}/models/${model_name}"

    if [ -d "$model_dir" ]; then
        print_info "Model directory exists: $model_name"
    else
        mkdir -p "$model_dir"
        print_success "Created model directory: $model_name"
    fi
}

# Function to save training config
save_training_config() {
    local model_dir="${REPO_DIR}/models/${MODEL_NAME}"
    local config_file="${model_dir}/training_config.json"

    cat > "$config_file" << EOF
{
  "model_name": "${MODEL_NAME}",
  "base_model": "${BASE_MODEL}",
  "dataset": "${DATASET}",
  "task_type": "${TASK_TYPE:-causal_lm}",
  "epochs": ${EPOCHS:-3},
  "batch_size": ${BATCH_SIZE:-8},
  "learning_rate": ${LEARNING_RATE:-2e-5},
  "created_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "created_from": "termux"
}
EOF

    print_success "Saved training config: $config_file"
}

# Function to commit and push changes
git_commit_push() {
    local message="$1"

    cd "$REPO_DIR"

    # Check if there are changes
    if [ -z "$(git status --porcelain)" ]; then
        print_info "No changes to commit"
        return 0
    fi

    print_info "Committing changes..."
    git add .
    git commit -m "$message"

    print_info "Pushing to GitHub..."
    git push

    print_success "Changes pushed to GitHub"
}

# Function to trigger GitHub Actions workflow
trigger_workflow() {
    print_info "Triggering GitHub Actions workflow..."

    # Get repository info
    local repo_url=$(git config --get remote.origin.url)
    local repo_name=$(echo "$repo_url" | sed -n 's#.*/\([^/]*\)/\([^/]*\)\.git#\1/\2#p')

    if [ -z "$repo_name" ]; then
        # Try alternative format
        repo_name=$(echo "$repo_url" | sed -n 's#.*/\([^/]*\)/\([^/]*\)$#\1/\2#p')
    fi

    if [ -z "$repo_name" ]; then
        print_error "Could not determine repository name"
        return 1
    fi

    # Trigger workflow using gh CLI if available
    if command -v gh &> /dev/null; then
        gh workflow run train.yml \
            -f model_name="$MODEL_NAME" \
            -f base_model="$BASE_MODEL" \
            -f dataset="$DATASET" \
            -f task_type="${TASK_TYPE:-causal_lm}" \
            -f epochs="${EPOCHS:-3}" \
            -f batch_size="${BATCH_SIZE:-8}" \
            -f push_to_hub="true"

        print_success "Workflow triggered successfully"
        print_info "Check status: gh run list"
    else
        print_warning "GitHub CLI (gh) not installed"
        print_info "Install with: pkg install gh"
        print_info "Or trigger manually on GitHub: https://github.com/${repo_name}/actions"
    fi
}

# Function to train locally (Termux)
train_local() {
    print_warning "Training locally on Termux (CPU only - this will be slow!)"
    print_info "Consider using Google Colab for faster training with GPU"

    cd "$REPO_DIR"

    # Check if Python is available
    if ! command -v python &> /dev/null; then
        print_error "Python not found. Install with: pkg install python"
        exit 1
    fi

    # Install dependencies if needed
    if ! python -c "import transformers" 2>/dev/null; then
        print_info "Installing dependencies..."
        pip install transformers datasets torch huggingface_hub
    fi

    # Build command
    local cmd="python train.py"
    cmd="$cmd --model_name $MODEL_NAME"
    cmd="$cmd --base_model $BASE_MODEL"
    cmd="$cmd --dataset $DATASET"
    cmd="$cmd --task_type ${TASK_TYPE:-causal_lm}"
    cmd="$cmd --epochs ${EPOCHS:-3}"
    cmd="$cmd --batch_size ${BATCH_SIZE:-8}"

    if [ -n "$HF_USERNAME" ]; then
        cmd="$cmd --hf_username $HF_USERNAME"
    fi

    if [ -n "$HF_TOKEN" ]; then
        cmd="$cmd --push_to_hub"
    fi

    print_info "Running: $cmd"
    eval $cmd
}

# Function to open Colab notebook
open_colab() {
    local repo_url=$(git config --get remote.origin.url)
    local repo_name=$(echo "$repo_url" | sed -n 's#.*/\([^/]*\)/\([^/]*\)\.git#\1/\2#p')

    if [ -z "$repo_name" ]; then
        repo_name=$(echo "$repo_url" | sed -n 's#.*/\([^/]*\)/\([^/]*\)$#\1/\2#p')
    fi

    local colab_url="https://colab.research.google.com/github/${repo_name}/blob/main/train_colab.ipynb"

    print_header "GOOGLE COLAB TRAINING"
    echo ""
    print_info "Open this URL in your browser:"
    echo ""
    echo "  $colab_url"
    echo ""
    print_info "In the Colab notebook, update the configuration:"
    echo "  - MODEL_NAME = \"$MODEL_NAME\""
    echo "  - BASE_MODEL = \"$BASE_MODEL\""
    echo "  - DATASET = \"$DATASET\""
    echo ""

    # Try to open in browser if termux-open-url is available
    if command -v termux-open-url &> /dev/null; then
        print_info "Opening in browser..."
        termux-open-url "$colab_url"
    fi
}

# Function to show help
show_help() {
    cat << EOF

Mobile LLM Lab - Termux Interface

USAGE:
  ./train_model.sh [command] [options]

COMMANDS:
  train              Start training with specified parameters
  local              Train locally on Termux (slow, CPU only)
  colab              Get Colab notebook link
  status             Show status of current models
  list               List all models
  config             Show current configuration
  help               Show this help message

SIMPLIFIED COMMAND FORMAT:
  ./train_model.sh 'train:model_name="assistant_v1", dataset="dataset/mydata.txt", base_model="distilbert-base-uncased"'

  Or with more options:
  ./train_model.sh 'train:model_name="assistant_v1", dataset="dataset/mydata.txt", base_model="distilbert-base-uncased", epochs=5, batch_size=16'

STANDARD COMMAND FORMAT:
  ./train_model.sh train \\
    --model_name "assistant_v1" \\
    --base_model "distilbert-base-uncased" \\
    --dataset "dataset/mydata.txt" \\
    --task_type "causal_lm" \\
    --epochs 3 \\
    --batch_size 8

PARAMETERS:
  --model_name       Name for your fine-tuned model (required)
  --base_model       Base model from Hugging Face (required)
  --dataset          Path to dataset file (required)
  --task_type        Task type: causal_lm or classification (default: causal_lm)
  --epochs           Number of training epochs (default: 3)
  --batch_size       Training batch size (default: 8)
  --learning_rate    Learning rate (default: 2e-5)

ENVIRONMENT VARIABLES:
  HF_TOKEN           Hugging Face access token (required for pushing models)
  HF_USERNAME        Hugging Face username (required for pushing models)
  GH_TOKEN           GitHub token (optional, for triggering workflows)

EXAMPLES:
  # Simple format
  ./train_model.sh 'train:model_name="my_assistant", dataset="dataset/data.txt", base_model="gpt2"'

  # Standard format
  ./train_model.sh train --model_name "my_assistant" --base_model "gpt2" --dataset "dataset/data.txt"

  # Train locally (slow)
  ./train_model.sh local --model_name "my_assistant" --base_model "gpt2" --dataset "dataset/data.txt"

  # Get Colab link
  ./train_model.sh colab

  # Check status
  ./train_model.sh status

EOF
}

# Function to show status
show_status() {
    print_header "MOBILE LLM LAB - STATUS"
    echo ""

    # Check tokens
    print_info "Environment Configuration:"
    if [ -n "$HF_TOKEN" ]; then
        print_success "HF_TOKEN is set"
    else
        print_error "HF_TOKEN not set"
    fi

    if [ -n "$HF_USERNAME" ]; then
        print_success "HF_USERNAME is set: $HF_USERNAME"
    else
        print_error "HF_USERNAME not set"
    fi

    if [ -n "$GH_TOKEN" ]; then
        print_success "GH_TOKEN is set"
    else
        print_warning "GH_TOKEN not set (optional)"
    fi

    echo ""
    print_info "Repository Information:"
    cd "$REPO_DIR"
    local repo_url=$(git config --get remote.origin.url)
    echo "  Repository: $repo_url"
    echo "  Branch: $(git branch --show-current)"
    echo "  Last commit: $(git log -1 --format='%h - %s')"

    echo ""
    print_info "Models:"
    if [ -d "models" ] && [ "$(ls -A models)" ]; then
        for model_dir in models/*; do
            if [ -d "$model_dir" ]; then
                local model_name=$(basename "$model_dir")
                echo "  - $model_name"
                if [ -f "$model_dir/training_config.json" ]; then
                    echo "    Config: ✓"
                fi
                if [ -f "$model_dir/config.json" ]; then
                    echo "    Trained: ✓"
                fi
            fi
        done
    else
        echo "  No models yet"
    fi

    echo ""
    print_info "Datasets:"
    if [ -d "dataset" ] && [ "$(ls -A dataset)" ]; then
        for dataset_file in dataset/*; do
            if [ -f "$dataset_file" ]; then
                local filename=$(basename "$dataset_file")
                local lines=$(wc -l < "$dataset_file")
                echo "  - $filename ($lines lines)"
            fi
        done
    else
        echo "  No datasets yet"
    fi

    echo ""
}

# Function to list models
list_models() {
    print_header "AVAILABLE MODELS"
    echo ""

    if [ -d "models" ] && [ "$(ls -A models)" ]; then
        for model_dir in models/*; do
            if [ -d "$model_dir" ]; then
                local model_name=$(basename "$model_dir")
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                echo "Model: $model_name"

                if [ -f "$model_dir/training_config.json" ]; then
                    echo ""
                    echo "Configuration:"
                    cat "$model_dir/training_config.json" | grep -E '"(base_model|dataset|epochs|batch_size)"' | sed 's/^/  /'
                fi

                if [ -f "$model_dir/training_metrics.json" ]; then
                    echo ""
                    echo "Metrics:"
                    cat "$model_dir/training_metrics.json" | head -5 | sed 's/^/  /'
                fi

                echo ""
            fi
        done
    else
        print_info "No models found"
    fi
}

# Main command router
main() {
    cd "$REPO_DIR"

    # If no arguments, show help
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi

    local command="$1"
    shift

    # Handle simplified command format: train:key=value,key=value
    if [[ "$command" == train:* ]]; then
        print_header "MOBILE LLM LAB - TRAINING"

        # Parse command
        parse_simple_command "$command"

        # Validate required parameters
        if [ -z "$MODEL_NAME" ] || [ -z "$BASE_MODEL" ] || [ -z "$DATASET" ]; then
            print_error "Missing required parameters"
            echo "Required: model_name, base_model, dataset"
            exit 1
        fi

        # Show configuration
        echo ""
        print_info "Configuration:"
        echo "  Model Name: $MODEL_NAME"
        echo "  Base Model: $BASE_MODEL"
        echo "  Dataset: $DATASET"
        echo "  Task Type: ${TASK_TYPE:-causal_lm}"
        echo "  Epochs: ${EPOCHS:-3}"
        echo "  Batch Size: ${BATCH_SIZE:-8}"
        echo ""

        # Check if model exists on HF
        if [ -n "$HF_USERNAME" ]; then
            check_hf_model_exists "$MODEL_NAME" "$HF_USERNAME"
        fi

        # Create model directory
        create_model_dir "$MODEL_NAME"

        # Save config
        save_training_config

        # Commit and push
        git_commit_push "Setup training for ${MODEL_NAME}"

        # Trigger workflow
        trigger_workflow

        # Show Colab link
        echo ""
        open_colab

        exit 0
    fi

    # Handle standard commands
    case "$command" in
        train)
            # Parse standard arguments
            while [ $# -gt 0 ]; do
                case "$1" in
                    --model_name) MODEL_NAME="$2"; shift 2 ;;
                    --base_model) BASE_MODEL="$2"; shift 2 ;;
                    --dataset) DATASET="$2"; shift 2 ;;
                    --task_type) TASK_TYPE="$2"; shift 2 ;;
                    --epochs) EPOCHS="$2"; shift 2 ;;
                    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
                    --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
                    *) print_warning "Unknown option: $1"; shift ;;
                esac
            done

            # Same logic as simplified format
            if [ -z "$MODEL_NAME" ] || [ -z "$BASE_MODEL" ] || [ -z "$DATASET" ]; then
                print_error "Missing required parameters"
                echo "Required: --model_name, --base_model, --dataset"
                exit 1
            fi

            print_header "MOBILE LLM LAB - TRAINING"

            create_model_dir "$MODEL_NAME"
            save_training_config
            git_commit_push "Setup training for ${MODEL_NAME}"
            trigger_workflow
            open_colab
            ;;

        local)
            # Parse arguments
            while [ $# -gt 0 ]; do
                case "$1" in
                    --model_name) MODEL_NAME="$2"; shift 2 ;;
                    --base_model) BASE_MODEL="$2"; shift 2 ;;
                    --dataset) DATASET="$2"; shift 2 ;;
                    --task_type) TASK_TYPE="$2"; shift 2 ;;
                    --epochs) EPOCHS="$2"; shift 2 ;;
                    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
                    --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
                    *) print_warning "Unknown option: $1"; shift ;;
                esac
            done

            if [ -z "$MODEL_NAME" ] || [ -z "$BASE_MODEL" ] || [ -z "$DATASET" ]; then
                print_error "Missing required parameters"
                exit 1
            fi

            print_header "MOBILE LLM LAB - LOCAL TRAINING"

            create_model_dir "$MODEL_NAME"
            save_training_config
            train_local
            ;;

        colab)
            open_colab
            ;;

        status)
            show_status
            ;;

        list)
            list_models
            ;;

        config)
            print_header "CONFIGURATION"
            echo ""
            echo "HF_TOKEN: ${HF_TOKEN:+set}"
            echo "HF_USERNAME: ${HF_USERNAME:-not set}"
            echo "GH_TOKEN: ${GH_TOKEN:+set}"
            echo ""
            ;;

        help|--help|-h)
            show_help
            ;;

        *)
            print_error "Unknown command: $command"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
