#!/bin/bash

################################################################
#################### For both Unseen and seen ##################
################################################################


##################################################################
###################### MULTI-METHOD EVALUATION #################
##################################################################

# --- 1. CONFIGURATION ---
# Define all artifact paths with their seeds
declare -A ARTIFACT_PATHS=(
    # ["Modumorph_seed1"]="artifacts/Robosuite-v0-MR-ST-MR-MT_ModuMorph_1409-run:v5"
    # ["Modumorph_seed2"]="artifacts/Robosuite-v0-MR-ST-MR-MT_ModuMorph_3296-run:v5"
    # ["Modumorph_seed3"]="artifacts/Robosuite-v0-MR-ST-MR-MT_ModuMorph_3296-run:v5"
    ["allnodes_seed1"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_ALLNODES4_1409-run:v9"
    ["allnodes_seed2"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_ALLNODES4_3296-run:v9"
    ["allnodes_seed3"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_ALLNODES4_2008-run:v9"
    ["avg_nodes_seed1"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_avg_nodes_1409-run:v19"
    ["avg_nodes_seed2"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_avg_nodes_3296-run:v18"
    ["avg_nodes_seed3"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_avg_nodes_2008-run:v18"
    # ["object_only_seed1"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_object_only_1409-run:v20"
    # ["object_only_seed2"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_object_only_3296-run:v18"
    # ["object_only_seed3"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_object_only4_2008-run:v9"
)

# Define checkpoint file (assuming same for all)
CHECKPOINT="checkpoint_600.pt"

# Define all morphologies to test
MORPHS=("Jaco" "Kinova3" "IIWA" "Sawyer")

# Define all tasks to test  
TASKS=("LiftBall" "LiftCylinder")

# The BASE task from which the learned skill/embedding should be drawn
BASE_TASK="Lift"

# Controller used for this robot-task pair
CONTROLLER="OSC_POSE"

# Number of episodes to run for each scaling factor
EPISODES=10

# Directory to save videos (optional, can be left empty)
VIDEO_DIR="./analysis_results/multi_eval_videos/"

# Define the percentage increases you want to test
PERCENT_INCREASES=(0 20 30 50 60 80 90 100 150 200)

# --- END OF CONFIGURATION ---

# --- 2. SCRIPT LOGIC ---

# Create results directory
RESULTS_DIR="./multi_eval_results"
mkdir -p $RESULTS_DIR

# Create main aggregated results file
AGGREGATED_RESULTS="$RESULTS_DIR/aggregated_results.csv"
echo "Method,Morph,Task,Percentage_Increase,Scale_Factor,Success_Rate_Mean,Success_Rate_Std,Avg_Reward_Mean,Avg_Reward_Std" > $AGGREGATED_RESULTS

echo "======================================================"
echo "MULTI-METHOD OBJECT GENERALIZATION EVALUATION"
echo "Methods: allnodes, avg_nodes, object_only (3 seeds each)"
echo "Morphs: ${MORPHS[*]}"
echo "Tasks: ${TASKS[*]}"
echo "Scale factors: ${PERCENT_INCREASES[*]}%"
echo "Results will be aggregated and saved to: $AGGREGATED_RESULTS"
echo "======================================================"

# Function to extract values from the Python script's final summary
extract_values() {
    local output="$1"
    
    # Extract success rate: "Success Rate: 60.0%"
    local success_rate=$(echo "$output" | grep -oP 'Success Rate:\s*\K[0-9.]+')
    
    # Extract average reward: "Avg. Reward: 255.36 ± 111.54"
    local avg_reward=$(echo "$output" | grep -oP 'Avg. Reward:\s*\K[0-9.]+')
    
    # Default to 0.0 if not found
    success_rate=${success_rate:-0.0}
    avg_reward=${avg_reward:-0.0}
    
    echo "$success_rate,$avg_reward"
}

# Get unique method names
declare -A METHODS
for key in "${!ARTIFACT_PATHS[@]}"; do
    method=$(echo $key | sed 's/_seed[0-9]*$//')
    METHODS[$method]=1
done

total_combinations=$((${#METHODS[@]} * ${#MORPHS[@]} * ${#TASKS[@]} * ${#PERCENT_INCREASES[@]}))
current_combo=0

# Loop through each method
for method in "${!METHODS[@]}"; do
    echo -e "\n🔬 Processing method: $method"
    
    # Loop through each morphology
    for morph in "${MORPHS[@]}"; do
        echo -e "\n  🤖 Testing morphology: $morph"
        
        # Loop through each task
        for task in "${TASKS[@]}"; do
            echo -e "\n    🎯 Testing task: $task"
            
            # Loop through each percentage increase
            for percent in "${PERCENT_INCREASES[@]}"; do
                current_combo=$((current_combo + 1))
                scale=$(echo "1 + $percent / 100" | bc -l)
                
                echo -e "\n      📊 Scale ${percent}% (${scale}) - Progress: ${current_combo}/${total_combinations}"
                
                # Collect results from all 3 seeds for this method
                declare -a success_rates=()
                declare -a avg_rewards=()
                
                # Loop through seeds 1, 2, 3 for current method
                for seed_num in 1 2 3; do
                    key="${method}_seed${seed_num}"
                    run_dir="${ARTIFACT_PATHS[$key]}"
                    
                    if [[ -z "$run_dir" ]]; then
                        echo "        ⚠️  Warning: No artifact path for ${key}, skipping..."
                        continue
                    fi
                    
                    echo "        🔄 Running seed ${seed_num}: ${key}"
                    
                    # Extract seed from run_dir for reproducibility
                    seed=$(echo $run_dir | grep -oP '[0-9]+(?=-run)' | tail -1)
                    seed=${seed:-$seed_num}
                    
                    # Construct the command
                    CMD="python tools/obj_geom_seeded.py \
                      --run_dir $run_dir \
                      --checkpoint $CHECKPOINT \
                      --morph $morph \
                      --task $task \
                      --base_task $BASE_TASK \
                      --controller $CONTROLLER \
                      --episodes $EPISODES \
                      --save_video $VIDEO_DIR \
                      --scale $scale \
                      --seed $seed"

                    # Execute the command and capture output
                    output=$($CMD 2>&1)
                    
                    # Extract results from the output
                    values=$(extract_values "$output")
                    success_rate=$(echo $values | cut -d',' -f1)
                    avg_reward=$(echo $values | cut -d',' -f2)
                    
                    # Store results
                    success_rates+=($success_rate)
                    avg_rewards+=($avg_reward)
                    
                    echo "        ✅ Seed ${seed_num}: Success=${success_rate}%, Reward=${avg_reward}"
                done
                
                # Calculate aggregated statistics
                if [ ${#success_rates[@]} -gt 0 ]; then
                    # Convert arrays to comma-separated strings for Python
                    success_str=$(IFS=','; echo "${success_rates[*]}")
                    reward_str=$(IFS=','; echo "${avg_rewards[*]}")
                    
                    # Calculate mean and std for success rates
                    success_mean=$(python3 -c "
import numpy as np
data = [$success_str]
print(f'{np.mean(data):.2f}')
")
                    success_std=$(python3 -c "
import numpy as np
data = [$success_str]
print(f'{np.std(data):.2f}')
")
                    
                    # Calculate mean and std for avg rewards  
                    reward_mean=$(python3 -c "
import numpy as np
data = [$reward_str]
print(f'{np.mean(data):.2f}')
")
                    reward_std=$(python3 -c "
import numpy as np
data = [$reward_str]
print(f'{np.std(data):.2f}')
")
                    
                    # Save aggregated results to main CSV
                    echo "$method,$morph,$task,$percent,$scale,$success_mean,$success_std,$reward_mean,$reward_std" >> $AGGREGATED_RESULTS
                    
                    echo "      📈 Aggregated - Success: ${success_mean}±${success_std}%, Reward: ${reward_mean}±${reward_std}"
                    
                    # Save individual seed results for this combination
                    individual_file="$RESULTS_DIR/individual_${method}_${morph}_${task}_${percent}pct.csv"
                    echo "Seed,Key,Success_Rate,Avg_Reward" > $individual_file
                    seed_idx=0
                    for key in ${seed_keys[@]}; do
                        if [ $seed_idx -lt ${#success_rates[@]} ]; then
                            echo "$((seed_idx+1)),$key,${success_rates[$seed_idx]},${avg_rewards[$seed_idx]}" >> $individual_file
                        fi
                        seed_idx=$((seed_idx + 1))
                    done
                else
                    echo "      ❌ No valid results for this combination"
                    echo "$method,$morph,$task,$percent,$scale,0.0,0.0,0.0,0.0" >> $AGGREGATED_RESULTS
                fi
                
                # Clear arrays for next iteration
                unset success_rates
                unset avg_rewards
                declare -a success_rates=()
                declare -a avg_rewards=()
            done
        done
    done
done

echo -e "\n======================================================"
echo "🎉 MULTI-METHOD EVALUATION COMPLETE!"
echo "📊 Aggregated results saved to: $AGGREGATED_RESULTS"
echo "📁 Individual seed results saved to: $RESULTS_DIR/individual_*.csv"
echo "======================================================"

# Display the final aggregated results
echo -e "\n📋 Final Aggregated Results:"
echo "============================================="
cat $AGGREGATED_RESULTS

# Generate summary plots
echo -e "\n📈 Generating summary plots..."
python3 << EOF
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

try:
    # Read the aggregated results
    df = pd.read_csv("$AGGREGATED_RESULTS")
    
    print("\\n📊 Aggregated data loaded successfully:")
    print(f"Shape: {df.shape}")
    print("\\nFirst few rows:")
    print(df.head(10))
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a comprehensive figure
    methods = df['Method'].unique()
    morphs = df['Morph'].unique()
    tasks = df['Task'].unique()
    
    # Plot 1: Success rate heatmap by method and scale
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Success rate by method and scale (averaged across all morph-task combinations)
    success_pivot = df.groupby(['Method', 'Percentage_Increase'])['Success_Rate_Mean'].mean().unstack()
    sns.heatmap(success_pivot, annot=True, fmt='.1f', ax=axes[0,0], cmap='RdYlGn')
    axes[0,0].set_title('Average Success Rate by Method and Scale (%)')
    axes[0,0].set_ylabel('Method')
    
    # Reward by method and scale
    reward_pivot = df.groupby(['Method', 'Percentage_Increase'])['Avg_Reward_Mean'].mean().unstack()
    sns.heatmap(reward_pivot, annot=True, fmt='.2f', ax=axes[0,1], cmap='viridis')
    axes[0,1].set_title('Average Reward by Method and Scale')
    axes[0,1].set_ylabel('Method')
    
    # Success rate trends by method
    for method in methods:
        method_data = df[df['Method'] == method].groupby('Percentage_Increase')['Success_Rate_Mean'].mean()
        axes[1,0].plot(method_data.index, method_data.values, 'o-', label=method, linewidth=2, markersize=6)
    
    axes[1,0].set_xlabel('Object Size Increase (%)')
    axes[1,0].set_ylabel('Success Rate (%)')
    axes[1,0].set_title('Success Rate Trends by Method')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Reward trends by method  
    for method in methods:
        method_data = df[df['Method'] == method].groupby('Percentage_Increase')['Avg_Reward_Mean'].mean()
        axes[1,1].plot(method_data.index, method_data.values, 's-', label=method, linewidth=2, markersize=6)
    
    axes[1,1].set_xlabel('Object Size Increase (%)')
    axes[1,1].set_ylabel('Average Reward')
    axes[1,1].set_title('Reward Trends by Method')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.suptitle('Multi-Method Object Generalization Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('$RESULTS_DIR/multi_method_analysis.png', dpi=150, bbox_inches='tight')
    print("✅ Main analysis plot saved to '$RESULTS_DIR/multi_method_analysis.png'")
    
    # Create detailed breakdown by morph and task
    if len(morphs) > 1 and len(tasks) > 1:
        fig, axes = plt.subplots(len(tasks), len(morphs), figsize=(4*len(morphs), 4*len(tasks)), sharey=True)
        if len(tasks) == 1:
            axes = axes.reshape(1, -1)
        if len(morphs) == 1:
            axes = axes.reshape(-1, 1)
            
        for i, task in enumerate(tasks):
            for j, morph in enumerate(morphs):
                subset = df[(df['Task'] == task) & (df['Morph'] == morph)]
                
                for method in methods:
                    method_data = subset[subset['Method'] == method]
                    if not method_data.empty:
                        axes[i,j].errorbar(method_data['Percentage_Increase'], 
                                         method_data['Success_Rate_Mean'],
                                         yerr=method_data['Success_Rate_Std'],
                                         label=method, marker='o', capsize=4)
                
                axes[i,j].set_title(f'{morph} - {task}')
                axes[i,j].set_xlabel('Scale Increase (%)')
                if j == 0:
                    axes[i,j].set_ylabel('Success Rate (%)')
                axes[i,j].legend()
                axes[i,j].grid(True, alpha=0.3)
        
        plt.suptitle('Success Rate by Morphology and Task', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('$RESULTS_DIR/detailed_breakdown.png', dpi=150, bbox_inches='tight')
        print("✅ Detailed breakdown plot saved to '$RESULTS_DIR/detailed_breakdown.png'")
    
    # Print summary statistics
    print("\\n📊 Summary Statistics:")
    print("="*60)
    for method in methods:
        method_data = df[df['Method'] == method]
        baseline_success = method_data[method_data['Percentage_Increase'] == 0]['Success_Rate_Mean'].mean()
        final_success = method_data[method_data['Percentage_Increase'] == method_data['Percentage_Increase'].max()]['Success_Rate_Mean'].mean()
        
        print(f"{method:12} - Baseline: {baseline_success:.1f}%, Final: {final_success:.1f}%, Drop: {baseline_success-final_success:.1f}%")
    
    print("="*60)
    print("\\n🎯 Best performing method at each scale:")
    for scale in sorted(df['Percentage_Increase'].unique()):
        scale_data = df[df['Percentage_Increase'] == scale]
        best_method = scale_data.loc[scale_data['Success_Rate_Mean'].idxmax(), 'Method']
        best_score = scale_data['Success_Rate_Mean'].max()
        print(f"  {scale:3.0f}% increase: {best_method} ({best_score:.1f}% success)")
    
except Exception as e:
    print(f"❌ Error creating plots: {e}")
    import traceback
    traceback.print_exc()

EOF

echo -e "\n✨ Analysis complete! Check the results directory: $RESULTS_DIR"