#!/bin/bash

# List available experiment directories for visualization

echo "📂 Available Experiment Directories:"
echo "====================================="

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
LOGS_DIR="$PROJECT_ROOT/logs"

if [ ! -d "$LOGS_DIR" ]; then
    echo "❌ No logs directory found at: $LOGS_DIR"
    exit 1
fi

for exp_dir in "$LOGS_DIR"/*; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename "$exp_dir")
        test_data="$exp_dir/test_env_data.json"
        checkpoint_dir="$exp_dir/checkpoints"
        
        echo ""
        echo "📁 $exp_name"
        
        if [ -f "$test_data" ]; then
            echo "   ✅ Test data: test_env_data.json"
        else
            echo "   ❌ Test data: MISSING"
        fi
        
        if [ -d "$checkpoint_dir" ] && [ -n "$(ls -A "$checkpoint_dir"/*.pt 2>/dev/null)" ]; then
            checkpoint_count=$(ls -1 "$checkpoint_dir"/*.pt 2>/dev/null | wc -l)
            echo "   ✅ Checkpoints: $checkpoint_count model(s)"
        else
            echo "   ❌ Checkpoints: MISSING"
        fi
        
        # Check if this directory is ready for visualization
        if [ -f "$test_data" ] && [ -d "$checkpoint_dir" ] && [ -n "$(ls -A "$checkpoint_dir"/*.pt 2>/dev/null)" ]; then
            echo "   🚀 Ready for visualization"
            echo "   📝 Usage: ./run_visualizer.sh logs/$exp_name"
        else
            echo "   ⚠️  Not ready (missing files)"
        fi
    fi
done

echo ""
echo "💡 To visualize an experiment:"
echo "   ./run_visualizer.sh logs/exp4"
echo "   ./run_visualizer.sh logs/exp5"
echo "   etc."