# PPO Training Makefile
# Simplified commands for PPO agent training and management

.PHONY: help install train test list clean resume quick-train long-train setup deps debug status benchmark check-deps info clean-all test-model

# Default settings
TIMESTEPS ?= 100000
SAVE_FREQ ?= 50000
TEST_EPISODES ?= 3
KEEP_MODELS ?= 5
MODEL ?= ppo_model_final.pth

# Python command (adjust if using different environment)
PYTHON ?= python

# Colors for output
GREEN = \033[0;32m
YELLOW = \033[1;33m
BLUE = \033[0;34m
RED = \033[0;31m
NC = \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)PPO Agent Training Makefile$(NC)"
	@echo "$(YELLOW)Available commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Variables:$(NC)"
	@echo "  TIMESTEPS=$(TIMESTEPS)"
	@echo "  SAVE_FREQ=$(SAVE_FREQ)"
	@echo "  TEST_EPISODES=$(TEST_EPISODES)"
	@echo "  KEEP_MODELS=$(KEEP_MODELS)"
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make train TIMESTEPS=50000        # Train for 50k timesteps"
	@echo "  make test TEST_EPISODES=5         # Test with 5 episodes"
	@echo "  make resume MODEL=ppo_model_50000.pth  # Resume from checkpoint"

setup: ## Setup Python environment and install dependencies
	@echo "$(BLUE)Setting up environment...$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)Setup complete!$(NC)"

install: setup ## Alias for setup

deps: ## Install only basic dependencies (torch, gym, numpy, matplotlib)
	@echo "$(BLUE)Installing basic dependencies...$(NC)"
	pip install torch gymnasium numpy matplotlib
	@echo "$(GREEN)Basic dependencies installed!$(NC)"

train: ## Train PPO agent from scratch
	@echo "$(BLUE)Starting training...$(NC)"
	@echo "$(YELLOW)Timesteps: $(TIMESTEPS), Save frequency: $(SAVE_FREQ)$(NC)"
	$(PYTHON) train.py --mode train --timesteps $(TIMESTEPS) --save_freq $(SAVE_FREQ)

quick-train: ## Quick training (10k timesteps, save every 2k)
	@echo "$(BLUE)Starting quick training...$(NC)"
	$(PYTHON) train.py --mode train --timesteps 10000 --save_freq 2000

long-train: ## Long training (1M timesteps, save every 50k)
	@echo "$(BLUE)Starting long training...$(NC)"
	$(PYTHON) train.py --mode train --timesteps 1000000 --save_freq 50000

resume: ## Resume training from a model checkpoint (use MODEL=filename)
	@echo "$(BLUE)Resuming training from $(MODEL)...$(NC)"
	$(PYTHON) train.py --mode train --resume_from $(MODEL) --timesteps 1000000 --save_freq $(SAVE_FREQ)

test: ## Test the trained agent
	@echo "$(BLUE)Testing agent for $(TEST_EPISODES) episodes...$(NC)"
	$(PYTHON) train.py --mode test --test_episodes $(TEST_EPISODES) --model_path $(MODEL)

test-model: ## Test a specific model (use MODEL=filename)
ifndef MODEL
	@echo "$(RED)Error: Please specify MODEL=filename$(NC)"
	@echo "$(YELLOW)Example: make test-model MODEL=ppo_model_final.pth$(NC)"
	@echo "$(YELLOW)Available models:$(NC)"
	@$(PYTHON) train.py --mode list
else
	@echo "$(BLUE)Testing model $(MODEL) for $(TEST_EPISODES) episodes...$(NC)"
	$(PYTHON) train.py --mode test --model_path $(MODEL) --test_episodes $(TEST_EPISODES)
endif

list: ## List all saved models for current environment
	@echo "$(BLUE)Saved models:$(NC)"
	$(PYTHON) train.py --mode list

clean: ## Clean old model files (keeps latest $(KEEP_MODELS) models)
	@echo "$(YELLOW)Cleaning old models (keeping $(KEEP_MODELS) latest)...$(NC)"
	$(PYTHON) train.py --mode clean --keep_models $(KEEP_MODELS)

clean-all: ## Remove all model files (WARNING: This deletes everything!)
	@echo "$(RED)WARNING: This will delete ALL model files!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "$(RED)Deleting all models...$(NC)"; \
		rm -rf models/; \
		echo "$(GREEN)All models deleted.$(NC)"; \
	else \
		echo "$(GREEN)Operation cancelled.$(NC)"; \
	fi

# Development and debugging
debug: ## Run training with debug output (short run)
	@echo "$(BLUE)Debug training run...$(NC)"
	$(PYTHON) train.py --mode train --timesteps 1000 --save_freq 500

status: ## Show training status and model information
	@echo "$(BLUE)=== PPO Training Status ===$(NC)"
	@echo "$(YELLOW)Available models:$(NC)"
	@$(PYTHON) train.py --mode list || echo "$(RED)No models found$(NC)"
	@echo ""
	@echo "$(YELLOW)Model directory structure:$(NC)"
	@if [ -d "models" ]; then \
		find models -name "*.pth" -exec ls -lh {} \; | head -10; \
		if [ $$(find models -name "*.pth" | wc -l) -gt 10 ]; then \
			echo "... and $$(expr $$(find models -name "*.pth" | wc -l) - 10) more files"; \
		fi \
	else \
		echo "$(RED)No models directory found$(NC)"; \
	fi

benchmark: ## Run a quick benchmark (train + test cycle)
	@echo "$(BLUE)Running benchmark (quick train + test)...$(NC)"
	$(MAKE) quick-train
	$(MAKE) test TEST_EPISODES=3

# Utility commands
check-deps: ## Check if required dependencies are installed
	@echo "$(BLUE)Checking dependencies...$(NC)"
	@$(PYTHON) -c "import torch; print('✓ PyTorch:', torch.__version__)" || echo "$(RED)✗ PyTorch not found$(NC)"
	@$(PYTHON) -c "import gymnasium; print('✓ Gymnasium:', gymnasium.__version__)" || echo "$(RED)✗ Gymnasium not found$(NC)"
	@$(PYTHON) -c "import numpy; print('✓ NumPy:', numpy.__version__)" || echo "$(RED)✗ NumPy not found$(NC)"
	@$(PYTHON) -c "import matplotlib; print('✓ Matplotlib:', matplotlib.__version__)" || echo "$(RED)✗ Matplotlib not found$(NC)"

info: ## Show system information
	@echo "$(BLUE)=== System Information ===$(NC)"
	@echo "$(YELLOW)Python version:$(NC)"
	@$(PYTHON) --version
	@echo "$(YELLOW)Current directory:$(NC) $$(pwd)"
	@echo "$(YELLOW)CUDA available:$(NC)"
	@$(PYTHON) -c "import torch; print('Yes' if torch.cuda.is_available() else 'No')"
	@echo "$(YELLOW)Environment config:$(NC) Set in config.py"

# Default target
.DEFAULT_GOAL := help