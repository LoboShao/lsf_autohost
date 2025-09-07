#!/bin/bash

# LSF Visualizer Interactive Launcher Script
# Usage: ./run_visualizer.sh [log_directory] [checkpoint_file]
# If no arguments provided, runs in interactive mode

LOG_DIR=${1}
CHECKPOINT_FILE=${2}

# Interactive mode if no arguments
if [ -z "$LOG_DIR" ]; then
    echo "LSF Visualizer - Interactive Mode"
    echo "================================="
    echo ""
    
    PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
    LOGS_DIR="$PROJECT_ROOT/logs"
    
    if [ ! -d "$LOGS_DIR" ]; then
        echo "ERROR: No logs directory found at: $LOGS_DIR"
        exit 1
    fi
    
    # Find valid experiment directories
    valid_experiments=()
    for exp_dir in "$LOGS_DIR"/*; do
        if [ -d "$exp_dir" ]; then
            exp_name=$(basename "$exp_dir")
            test_data="$exp_dir/test_env_data.json"
            checkpoint_dir="$exp_dir/checkpoints"
            
            if [ -f "$test_data" ] && [ -d "$checkpoint_dir" ] && [ -n "$(ls -A "$checkpoint_dir"/*.pt 2>/dev/null)" ]; then
                valid_experiments+=("logs/$exp_name")
            fi
        fi
    done
    
    if [ ${#valid_experiments[@]} -eq 0 ]; then
        echo "ERROR: No valid experiment directories found in $LOGS_DIR"
        echo "Each experiment needs test_env_data.json and checkpoints/*.pt"
        exit 1
    fi
    
    # Select experiment directory
    echo "Available experiment directories:"
    for i in "${!valid_experiments[@]}"; do
        echo "  $((i+1))) ${valid_experiments[$i]}"
    done
    echo ""
    
    while true; do
        read -p "Select experiment directory (1-${#valid_experiments[@]}): " selection
        if [[ "$selection" =~ ^[0-9]+$ ]] && [ "$selection" -ge 1 ] && [ "$selection" -le ${#valid_experiments[@]} ]; then
            LOG_DIR="${valid_experiments[$((selection-1))]}"
            break
        else
            echo "Invalid selection. Please enter a number between 1 and ${#valid_experiments[@]}."
        fi
    done
    
    echo ""
    echo "Selected: $LOG_DIR"
    echo ""
fi

echo "Starting LSF Visualizer for directory: $LOG_DIR"
echo "Data file: $LOG_DIR/test_env_data.json"
echo "Model checkpoints: $LOG_DIR/checkpoints/*.pt"
echo ""

# Check if required files exist
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATA_FILE="$PROJECT_ROOT/$LOG_DIR/test_env_data.json"
CHECKPOINT_DIR="$PROJECT_ROOT/$LOG_DIR/checkpoints"

if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: Test data file not found: $DATA_FILE"
    echo "Make sure you have run training and generated test data in this directory."
    exit 1
fi

if [ ! -d "$CHECKPOINT_DIR" ] || [ -z "$(ls -A "$CHECKPOINT_DIR"/*.pt 2>/dev/null)" ]; then
    echo "ERROR: No model checkpoints found in: $CHECKPOINT_DIR"
    echo "Make sure you have trained models with checkpoints saved."
    exit 1
fi

# Handle checkpoint selection
if [ -n "$CHECKPOINT_FILE" ]; then
    # User specified a checkpoint file
    CHECKPOINT_PATH="$CHECKPOINT_DIR/$CHECKPOINT_FILE"
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        echo "ERROR: Specified checkpoint not found: $CHECKPOINT_PATH"
        echo ""
        echo "Available checkpoints in $LOG_DIR/checkpoints:"
        ls -la "$CHECKPOINT_DIR"/*.pt 2>/dev/null | awk '{print "  " $9}' | sed 's|.*/||'
        exit 1
    fi
    echo "Using checkpoint: $CHECKPOINT_FILE"
else
    # Interactive checkpoint selection
    available_checkpoints=($(ls -t "$CHECKPOINT_DIR"/*.pt 2>/dev/null | sed 's|.*/||'))
    
    if [ ${#available_checkpoints[@]} -eq 0 ]; then
        echo "ERROR: No checkpoints found in $CHECKPOINT_DIR"
        exit 1
    fi
    
    echo "Available checkpoints (newest first):"
    for i in "${!available_checkpoints[@]}"; do
        echo "  $((i+1))) ${available_checkpoints[$i]}"
    done
    echo "  $((${#available_checkpoints[@]}+1))) Use latest checkpoint (auto)"
    echo ""
    
    while true; do
        read -p "Select checkpoint (1-$((${#available_checkpoints[@]}+1))): " selection
        if [[ "$selection" =~ ^[0-9]+$ ]]; then
            if [ "$selection" -eq $((${#available_checkpoints[@]}+1)) ]; then
                # Use latest (auto selection)
                CHECKPOINT_FILE="${available_checkpoints[0]}"
                echo "Auto-selected latest: $CHECKPOINT_FILE"
                break
            elif [ "$selection" -ge 1 ] && [ "$selection" -le ${#available_checkpoints[@]} ]; then
                CHECKPOINT_FILE="${available_checkpoints[$((selection-1))]}"
                echo "Selected: $CHECKPOINT_FILE"
                break
            fi
        fi
        echo "Invalid selection. Please enter a number between 1 and $((${#available_checkpoints[@]}+1))."
    done
fi

echo "Data files found. Starting dashboard..."
echo "Client: http://localhost:3000"
echo "Server: http://localhost:5001"
echo ""

# Export configuration for both server and client
export LSF_VISUALIZER_LOG_DIR="$LOG_DIR"           # For server (Node.js)
export REACT_APP_LOG_DIR="$LOG_DIR"                # For client (React)

if [ -n "$CHECKPOINT_FILE" ]; then
    export LSF_VISUALIZER_CHECKPOINT="$CHECKPOINT_FILE"  # For server checkpoint selection
fi

echo "Using log directory: $LOG_DIR"
if [ -n "$CHECKPOINT_FILE" ]; then
    echo "Using checkpoint: $CHECKPOINT_FILE"
fi
echo "Starting dashboard..."

# Start the dashboard
npm run dev