# #!/bin/bash

# ################################################################
# #################### For both Unseen and seen ##################
# ################################################################


# ##################################################################
# ###################### MULTI-METHOD EVALUATION #################
# ##################################################################

# # --- 1. CONFIGURATION ---
# # Define all artifact paths with their seeds
# declare -A ARTIFACT_PATHS=(
#     # ["Modumorph_seed1"]="artifacts/Robosuite-v0-MR-ST-MR-MT_ModuMorph_1409-run:v5"
#     # ["Modumorph_seed2"]="artifacts/Robosuite-v0-MR-ST-MR-MT_ModuMorph_3296-run:v5"
#     # ["Modumorph_seed3"]="artifacts/Robosuite-v0-MR-ST-MR-MT_ModuMorph_3296-run:v5"
#     # ["allnodes_seed1"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_ALLNODES4_1409-run:v9"
#     # ["allnodes_seed2"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_ALLNODES4_3296-run:v9"
#     # ["allnodes_seed3"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_ALLNODES4_2008-run:v9"
#     ["avg_nodes_seed1"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_avg_nodes_1409-run:v19"
#     ["avg_nodes_seed2"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_avg_nodes_3296-run:v18"
#     ["avg_nodes_seed3"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_avg_nodes_2008-run:v18"
#     # ["object_only_seed1"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_object_only_1409-run:v20"
#     # ["object_only_seed2"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_object_only_3296-run:v18"
#     # ["object_only_seed3"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_object_only4_2008-run:v9"
# )

# # Define checkpoint file (assuming same for all)
# CHECKPOINT="checkpoint_600.pt"

# # Define all morphologies to test
# MORPHS=("Jaco" "Kinova3" "IIWA" "Sawyer")

# # Define all tasks to test  
# TASKS=("LiftBall" "LiftCylinder")

# # The BASE task from which the learned skill/embedding should be drawn
# BASE_TASK="Lift"

# # Controller used for this robot-task pair
# CONTROLLER="OSC_POSE"

# # Number of episodes to run for each scaling factor
# EPISODES=10

# # Directory to save videos (optional, can be left empty)
# VIDEO_DIR="./analysis_results/multi_eval_videos/"

# # Define the percentage increases you want to test
# PERCENT_INCREASES=(0 20 30 50 60 80 90 100 150 200)

# # --- END OF CONFIGURATION ---

# # --- 2. SCRIPT LOGIC ---

# # Create results directory
# RESULTS_DIR="./multi_eval_results"
# mkdir -p $RESULTS_DIR

# # Create main aggregated results file
# AGGREGATED_RESULTS="$RESULTS_DIR/aggregated_results.csv"
# echo "Method,Morph,Task,Percentage_Increase,Scale_Factor,Success_Rate_Mean,Success_Rate_Std,Avg_Reward_Mean,Avg_Reward_Std" > $AGGREGATED_RESULTS

# echo "======================================================"
# echo "MULTI-METHOD OBJECT GENERALIZATION EVALUATION"
# echo "Methods: allnodes, avg_nodes, object_only (3 seeds each)"
# echo "Morphs: ${MORPHS[*]}"
# echo "Tasks: ${TASKS[*]}"
# echo "Scale factors: ${PERCENT_INCREASES[*]}%"
# echo "Results will be aggregated and saved to: $AGGREGATED_RESULTS"
# echo "======================================================"

# # Function to extract values from the Python script's final summary
# extract_values() {
#     local output="$1"
    
#     # Extract success rate: "Success Rate: 60.0%"
#     local success_rate=$(echo "$output" | grep -oP 'Success Rate:\s*\K[0-9.]+')
    
#     # Extract average reward: "Avg. Reward: 255.36 ± 111.54"
#     local avg_reward=$(echo "$output" | grep -oP 'Avg. Reward:\s*\K[0-9.]+')
    
#     # Default to 0.0 if not found
#     success_rate=${success_rate:-0.0}
#     avg_reward=${avg_reward:-0.0}
    
#     echo "$success_rate,$avg_reward"
# }

# # Get unique method names
# declare -A METHODS
# for key in "${!ARTIFACT_PATHS[@]}"; do
#     method=$(echo $key | sed 's/_seed[0-9]*$//')
#     METHODS[$method]=1
# done

# total_combinations=$((${#METHODS[@]} * ${#MORPHS[@]} * ${#TASKS[@]} * ${#PERCENT_INCREASES[@]}))
# current_combo=0

# # Loop through each method
# for method in "${!METHODS[@]}"; do
#     echo -e "\n🔬 Processing method: $method"
    
#     # Loop through each morphology
#     for morph in "${MORPHS[@]}"; do
#         echo -e "\n  🤖 Testing morphology: $morph"
        
#         # Loop through each task
#         for task in "${TASKS[@]}"; do
#             echo -e "\n    🎯 Testing task: $task"
            
#             # Loop through each percentage increase
#             for percent in "${PERCENT_INCREASES[@]}"; do
#                 current_combo=$((current_combo + 1))
#                 scale=$(echo "1 + $percent / 100" | bc -l)
                
#                 echo -e "\n      📊 Scale ${percent}% (${scale}) - Progress: ${current_combo}/${total_combinations}"
                
#                 # Collect results from all 3 seeds for this method
#                 declare -a success_rates=()
#                 declare -a avg_rewards=()
                
#                 # Loop through seeds 1, 2, 3 for current method
#                 for seed_num in 1 2 3; do
#                     key="${method}_seed${seed_num}"
#                     run_dir="${ARTIFACT_PATHS[$key]}"
                    
#                     if [[ -z "$run_dir" ]]; then
#                         echo "        ⚠️  Warning: No artifact path for ${key}, skipping..."
#                         continue
#                     fi
                    
#                     echo "        🔄 Running seed ${seed_num}: ${key}"
                    
#                     # Extract seed from run_dir for reproducibility
#                     seed=$(echo $run_dir | grep -oP '[0-9]+(?=-run)' | tail -1)
#                     seed=${seed:-$seed_num}
                    
#                     # Construct the command
#                     CMD="python tools/obj_geom_seeded.py \
#                       --run_dir $run_dir \
#                       --checkpoint $CHECKPOINT \
#                       --morph $morph \
#                       --task $task \
#                       --base_task $BASE_TASK \
#                       --controller $CONTROLLER \
#                       --episodes $EPISODES \
#                       --save_video $VIDEO_DIR \
#                       --scale $scale \
#                       --seed $seed"

#                     # Execute the command and capture output
#                     output=$($CMD 2>&1)
                    
#                     # Extract results from the output
#                     values=$(extract_values "$output")
#                     success_rate=$(echo $values | cut -d',' -f1)
#                     avg_reward=$(echo $values | cut -d',' -f2)
                    
#                     # Store results
#                     success_rates+=($success_rate)
#                     avg_rewards+=($avg_reward)
                    
#                     echo "        ✅ Seed ${seed_num}: Success=${success_rate}%, Reward=${avg_reward}"
#                 done
                
#                 # Calculate aggregated statistics
#                 if [ ${#success_rates[@]} -gt 0 ]; then
#                     # Convert arrays to comma-separated strings for Python
#                     success_str=$(IFS=','; echo "${success_rates[*]}")
#                     reward_str=$(IFS=','; echo "${avg_rewards[*]}")
                    
#                     # Calculate mean and std for success rates
#                     success_mean=$(python3 -c "
# import numpy as np
# data = [$success_str]
# print(f'{np.mean(data):.2f}')
# ")
#                     success_std=$(python3 -c "
# import numpy as np
# data = [$success_str]
# print(f'{np.std(data):.2f}')
# ")
                    
#                     # Calculate mean and std for avg rewards  
#                     reward_mean=$(python3 -c "
# import numpy as np
# data = [$reward_str]
# print(f'{np.mean(data):.2f}')
# ")
#                     reward_std=$(python3 -c "
# import numpy as np
# data = [$reward_str]
# print(f'{np.std(data):.2f}')
# ")
                    
#                     # Save aggregated results to main CSV
#                     echo "$method,$morph,$task,$percent,$scale,$success_mean,$success_std,$reward_mean,$reward_std" >> $AGGREGATED_RESULTS
                    
#                     echo "      📈 Aggregated - Success: ${success_mean}±${success_std}%, Reward: ${reward_mean}±${reward_std}"
                    
#                     # Save individual seed results for this combination
#                     individual_file="$RESULTS_DIR/individual_${method}_${morph}_${task}_${percent}pct.csv"
#                     echo "Seed,Key,Success_Rate,Avg_Reward" > $individual_file
#                     seed_idx=0
#                     for key in ${seed_keys[@]}; do
#                         if [ $seed_idx -lt ${#success_rates[@]} ]; then
#                             echo "$((seed_idx+1)),$key,${success_rates[$seed_idx]},${avg_rewards[$seed_idx]}" >> $individual_file
#                         fi
#                         seed_idx=$((seed_idx + 1))
#                     done
#                 else
#                     echo "      ❌ No valid results for this combination"
#                     echo "$method,$morph,$task,$percent,$scale,0.0,0.0,0.0,0.0" >> $AGGREGATED_RESULTS
#                 fi
                
#                 # Clear arrays for next iteration
#                 unset success_rates
#                 unset avg_rewards
#                 declare -a success_rates=()
#                 declare -a avg_rewards=()
#             done
#         done
#     done
# done

# echo -e "\n======================================================"
# echo "🎉 MULTI-METHOD EVALUATION COMPLETE!"
# echo "📊 Aggregated results saved to: $AGGREGATED_RESULTS"
# echo "📁 Individual seed results saved to: $RESULTS_DIR/individual_*.csv"
# echo "======================================================"

# # Display the final aggregated results
# echo -e "\n📋 Final Aggregated Results:"
# echo "============================================="
# cat $AGGREGATED_RESULTS

# # Generate summary plots
# echo -e "\n📈 Generating summary plots..."
# python3 << EOF
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import seaborn as sns

# try:
#     # Read the aggregated results
#     df = pd.read_csv("$AGGREGATED_RESULTS")
    
#     print("\\n📊 Aggregated data loaded successfully:")
#     print(f"Shape: {df.shape}")
#     print("\\nFirst few rows:")
#     print(df.head(10))
    
#     # Set up the plotting style
#     plt.style.use('default')
#     sns.set_palette("husl")
    
#     # Create a comprehensive figure
#     methods = df['Method'].unique()
#     morphs = df['Morph'].unique()
#     tasks = df['Task'].unique()
    
#     # Plot 1: Success rate heatmap by method and scale
#     fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
#     # Success rate by method and scale (averaged across all morph-task combinations)
#     success_pivot = df.groupby(['Method', 'Percentage_Increase'])['Success_Rate_Mean'].mean().unstack()
#     sns.heatmap(success_pivot, annot=True, fmt='.1f', ax=axes[0,0], cmap='RdYlGn')
#     axes[0,0].set_title('Average Success Rate by Method and Scale (%)')
#     axes[0,0].set_ylabel('Method')
    
#     # Reward by method and scale
#     reward_pivot = df.groupby(['Method', 'Percentage_Increase'])['Avg_Reward_Mean'].mean().unstack()
#     sns.heatmap(reward_pivot, annot=True, fmt='.2f', ax=axes[0,1], cmap='viridis')
#     axes[0,1].set_title('Average Reward by Method and Scale')
#     axes[0,1].set_ylabel('Method')
    
#     # Success rate trends by method
#     for method in methods:
#         method_data = df[df['Method'] == method].groupby('Percentage_Increase')['Success_Rate_Mean'].mean()
#         axes[1,0].plot(method_data.index, method_data.values, 'o-', label=method, linewidth=2, markersize=6)
    
#     axes[1,0].set_xlabel('Object Size Increase (%)')
#     axes[1,0].set_ylabel('Success Rate (%)')
#     axes[1,0].set_title('Success Rate Trends by Method')
#     axes[1,0].legend()
#     axes[1,0].grid(True, alpha=0.3)
    
#     # Reward trends by method  
#     for method in methods:
#         method_data = df[df['Method'] == method].groupby('Percentage_Increase')['Avg_Reward_Mean'].mean()
#         axes[1,1].plot(method_data.index, method_data.values, 's-', label=method, linewidth=2, markersize=6)
    
#     axes[1,1].set_xlabel('Object Size Increase (%)')
#     axes[1,1].set_ylabel('Average Reward')
#     axes[1,1].set_title('Reward Trends by Method')
#     axes[1,1].legend()
#     axes[1,1].grid(True, alpha=0.3)
    
#     plt.suptitle('Multi-Method Object Generalization Analysis', fontsize=16, fontweight='bold')
#     plt.tight_layout()
#     plt.savefig('$RESULTS_DIR/multi_method_analysis.png', dpi=150, bbox_inches='tight')
#     print("✅ Main analysis plot saved to '$RESULTS_DIR/multi_method_analysis.png'")
    
#     # Create detailed breakdown by morph and task
#     if len(morphs) > 1 and len(tasks) > 1:
#         fig, axes = plt.subplots(len(tasks), len(morphs), figsize=(4*len(morphs), 4*len(tasks)), sharey=True)
#         if len(tasks) == 1:
#             axes = axes.reshape(1, -1)
#         if len(morphs) == 1:
#             axes = axes.reshape(-1, 1)
            
#         for i, task in enumerate(tasks):
#             for j, morph in enumerate(morphs):
#                 subset = df[(df['Task'] == task) & (df['Morph'] == morph)]
                
#                 for method in methods:
#                     method_data = subset[subset['Method'] == method]
#                     if not method_data.empty:
#                         axes[i,j].errorbar(method_data['Percentage_Increase'], 
#                                          method_data['Success_Rate_Mean'],
#                                          yerr=method_data['Success_Rate_Std'],
#                                          label=method, marker='o', capsize=4)
                
#                 axes[i,j].set_title(f'{morph} - {task}')
#                 axes[i,j].set_xlabel('Scale Increase (%)')
#                 if j == 0:
#                     axes[i,j].set_ylabel('Success Rate (%)')
#                 axes[i,j].legend()
#                 axes[i,j].grid(True, alpha=0.3)
        
#         plt.suptitle('Success Rate by Morphology and Task', fontsize=16, fontweight='bold')
#         plt.tight_layout()
#         plt.savefig('$RESULTS_DIR/detailed_breakdown.png', dpi=150, bbox_inches='tight')
#         print("✅ Detailed breakdown plot saved to '$RESULTS_DIR/detailed_breakdown.png'")
    
#     # Print summary statistics
#     print("\\n📊 Summary Statistics:")
#     print("="*60)
#     for method in methods:
#         method_data = df[df['Method'] == method]
#         baseline_success = method_data[method_data['Percentage_Increase'] == 0]['Success_Rate_Mean'].mean()
#         final_success = method_data[method_data['Percentage_Increase'] == method_data['Percentage_Increase'].max()]['Success_Rate_Mean'].mean()
        
#         print(f"{method:12} - Baseline: {baseline_success:.1f}%, Final: {final_success:.1f}%, Drop: {baseline_success-final_success:.1f}%")
    
#     print("="*60)
#     print("\\n🎯 Best performing method at each scale:")
#     for scale in sorted(df['Percentage_Increase'].unique()):
#         scale_data = df[df['Percentage_Increase'] == scale]
#         best_method = scale_data.loc[scale_data['Success_Rate_Mean'].idxmax(), 'Method']
#         best_score = scale_data['Success_Rate_Mean'].max()
#         print(f"  {scale:3.0f}% increase: {best_method} ({best_score:.1f}% success)")
    
# except Exception as e:
#     print(f"❌ Error creating plots: {e}")
#     import traceback
#     traceback.print_exc()

# EOF

# echo -e "\n✨ Analysis complete! Check the results directory: $RESULTS_DIR"


# # cd ModuMorph && git checkout -- . && git pull origin spo && sudo apt-get install bc && sudo apt-get install ffmpeg -y && code down_wandb.py tools/object_generalization_auto.sh


#!/bin/bash

################################################################
#################### For both Unseen and seen ##################
################################################################

#!/bin/bash

################################################################
#################### PARALLEL MULTI-METHOD EVAL ##############
################################################################

# Check if GNU parallel is installed
if ! command -v parallel &> /dev/null; then
    echo "⚠️  GNU Parallel not found. Installing..."
    sudo apt-get update && sudo apt-get install -y parallel
    echo "✅ GNU Parallel installed successfully"
fi

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

# --- PARALLEL EXECUTION SETTINGS ---
# Adjust these based on your system resources
MAX_PARALLEL_JOBS=12        # Conservative: 16 cores / 2 (GPU+CPU intensive tasks)
GPU_JOBS_LIMIT=4          # Max simultaneous GPU jobs (L4 has limited VRAM)
MEMORY_PER_JOB="2G"       # Memory limit per job

# --- MULTI-TERMINAL SAFETY: Generate unique run identifier ---
RUN_ID="${USER}_$(hostname)_$$_$(date +%s%N | cut -c1-13)"
echo "🔒 Multi-terminal safety enabled. Run ID: $RUN_ID"
echo "⚡ Parallel execution: MAX_JOBS=$MAX_PARALLEL_JOBS, GPU_LIMIT=$GPU_JOBS_LIMIT"

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
EPISODES=5

# Directory to save videos (optional, can be left empty) - Now unique per run
VIDEO_DIR="./analysis_results/multi_eval_videos_${RUN_ID}/"

# Define the percentage increases you want to test
PERCENT_INCREASES=(0 20 30 50 60 80 90 100 150 200)

# --- END OF CONFIGURATION ---

# --- 2. SETUP ---

# Create results directory with unique identifier
RESULTS_DIR="./multi_eval_results_${RUN_ID}"
mkdir -p $RESULTS_DIR

# Create temporary directory for individual job results
TEMP_JOBS_DIR="$RESULTS_DIR/temp_jobs"
mkdir -p $TEMP_JOBS_DIR

# Create main aggregated results file with unique name
AGGREGATED_RESULTS="$RESULTS_DIR/aggregated_results_${RUN_ID}.csv"
echo "Method,Morph,Task,Percentage_Increase,Scale_Factor,Success_Rate_Mean,Success_Rate_Std,Avg_Reward_Mean,Avg_Reward_Std" > $AGGREGATED_RESULTS

echo "======================================================"
echo "⚡ PARALLEL MULTI-METHOD OBJECT GENERALIZATION EVALUATION"
echo "🔒 Run ID: $RUN_ID (prevents multi-terminal conflicts)"
echo "⚙️  Max parallel jobs: $MAX_PARALLEL_JOBS"
echo "🎮 GPU job limit: $GPU_JOBS_LIMIT" 
echo "💾 Memory per job: $MEMORY_PER_JOB"
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

# Function to run a single evaluation job
run_single_job() {
    local job_id="$1"
    local method="$2"
    local key="$3"
    local run_dir="$4"
    local morph="$5"
    local task="$6"
    local percent="$7"
    local scale="$8"
    local seed="$9"
    
    # Create job-specific output file
    local job_output_file="$TEMP_JOBS_DIR/job_${job_id}_${method}_${morph}_${task}_${percent}pct.txt"
    
    echo "[JOB-$job_id] Starting: $method $morph $task ${percent}% (seed $seed)" >> "$job_output_file"
    
    # Construct the command with unique video directory
    local CMD="python tools/obj_geom_seeded.py \
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
    local output
    if output=$($CMD 2>&1); then
        # Extract results from the output
        local values=$(extract_values "$output")
        local success_rate=$(echo $values | cut -d',' -f1)
        local avg_reward=$(echo $values | cut -d',' -f2)
        
        # Write results to job file
        echo "SUCCESS,$success_rate,$avg_reward" >> "$job_output_file"
        echo "[JOB-$job_id] ✅ Success=${success_rate}%, Reward=${avg_reward}" >> "$job_output_file"
    else
        # Job failed
        echo "FAILED,0.0,0.0" >> "$job_output_file"
        echo "[JOB-$job_id] ❌ FAILED" >> "$job_output_file"
        echo "Error output: $output" >> "$job_output_file"
    fi
}

# Export function so parallel can use it
export -f run_single_job
export -f extract_values
export TEMP_JOBS_DIR CHECKPOINT BASE_TASK CONTROLLER EPISODES VIDEO_DIR

# --- 3. GENERATE JOB LIST ---

# Get unique method names
declare -A METHODS
for key in "${!ARTIFACT_PATHS[@]}"; do
    method=$(echo $key | sed 's/_seed[0-9]*$//')
    METHODS[$method]=1
done

echo "🔄 Generating parallel job list..."

# Create job list file
JOB_LIST_FILE="$TEMP_JOBS_DIR/job_list.txt"
> $JOB_LIST_FILE  # Clear file

job_counter=0
total_combinations=$((${#METHODS[@]} * ${#MORPHS[@]} * ${#TASKS[@]} * ${#PERCENT_INCREASES[@]} * 3))

# Generate all job combinations
for method in "${!METHODS[@]}"; do
    for morph in "${MORPHS[@]}"; do
        for task in "${TASKS[@]}"; do
            for percent in "${PERCENT_INCREASES[@]}"; do
                scale=$(echo "1 + $percent / 100" | bc -l)
                
                # Loop through seeds 1, 2, 3 for current method
                for seed_num in 1 2 3; do
                    key="${method}_seed${seed_num}"
                    run_dir="${ARTIFACT_PATHS[$key]}"
                    
                    if [[ -z "$run_dir" ]]; then
                        echo "⚠️  Warning: No artifact path for ${key}, skipping..."
                        continue
                    fi
                    
                    # Extract seed from run_dir for reproducibility
                    seed=$(echo $run_dir | grep -oP '[0-9]+(?=-run)' | tail -1)
                    seed=${seed:-$seed_num}
                    
                    job_counter=$((job_counter + 1))
                    
                    # Add job to list: job_id method key run_dir morph task percent scale seed
                    echo "$job_counter $method $key $run_dir $morph $task $percent $scale $seed" >> $JOB_LIST_FILE
                done
            done
        done
    done
done

echo "📊 Generated $job_counter parallel jobs (estimated total: $total_combinations)"

# --- 4. EXECUTE JOBS IN PARALLEL ---

echo "⚡ Starting parallel execution..."
echo "📈 Progress can be monitored in: $TEMP_JOBS_DIR/"

# Use GNU parallel to execute jobs
# --jobs: limit concurrent jobs
# --colsep: use space as column separator  
# --progress: show progress bar
# --bar: show progress bar
# --eta: show estimated time
parallel \
    --jobs $MAX_PARALLEL_JOBS \
    --colsep ' ' \
    --progress \
    --bar \
    --eta \
    --joblog "$RESULTS_DIR/parallel_execution_log.txt" \
    --memfree $MEMORY_PER_JOB \
    run_single_job {1} {2} {3} {4} {5} {6} {7} {8} {9} :::: $JOB_LIST_FILE

echo "✅ All parallel jobs completed!"

# --- 5. AGGREGATE RESULTS ---

echo "📊 Aggregating results from parallel jobs..."

# Process results by method-morph-task-percent combination
for method in "${!METHODS[@]}"; do
    echo "🔬 Processing method: $method"
    
    for morph in "${MORPHS[@]}"; do
        for task in "${TASKS[@]}"; do
            for percent in "${PERCENT_INCREASES[@]}"; do
                scale=$(echo "1 + $percent / 100" | bc -l)
                
                # Collect results from all 3 seeds for this combination
                declare -a success_rates=()
                declare -a avg_rewards=()
                
                # Look for job result files matching this combination
                for seed_num in 1 2 3; do
                    # pattern="job_*_${method}_${morph}_${task}_${percent}pct.txt"
                    # job_files=$(find "$TEMP_JOBS_DIR" -name "$pattern" | grep "seed${seed_num}" || true)
                    # More specific pattern matching
                    job_files=$(find "$TEMP_JOBS_DIR" -name "job_*_${method}_${morph}_${task}_${percent}pct.txt" | head -3)
                    # if [[ -n "$job_files" ]]; then 
                    # Process each job file for this combination
                    for job_file in $job_files; do
                        # for job_file in $job_files; do
                        #     if [[ -f "$job_file" ]]; then
                            if [[ -f "$job_file" && $(basename "$job_file") == *"${method}_${morph}_${task}_${percent}pct.txt" ]]; then
                                    # Extract SUCCESS,success_rate,avg_reward line
                                    result_line=$(grep "^SUCCESS," "$job_file" 2>/dev/null || echo "FAILED,0.0,0.0")
                                    success_rate=$(echo $result_line | cut -d',' -f2)
                                    avg_reward=$(echo $result_line | cut -d',' -f3)
                                    
                                    success_rates+=($success_rate)
                                    avg_rewards+=($avg_reward)
                                    break  # Only take first matching file per seed
                            # fi
                        # done
                    # fi
                
                            fi
                    done 
                done
                
                # Calculate aggregated statistics
                if [ ${#success_rates[@]} -gt 0 ]; then
                    # Convert arrays to comma-separated strings for Python
                    success_str=$(IFS=','; echo "${success_rates[*]}")
                    reward_str=$(IFS=','; echo "${avg_rewards[*]}")
                    
                    # Calculate mean and std
                    stats=$(python3 -c "
import numpy as np
success_data = [$success_str]
reward_data = [$reward_str]
print(f'{np.mean(success_data):.2f},{np.std(success_data):.2f},{np.mean(reward_data):.2f},{np.std(reward_data):.2f}')
")
                    
                    success_mean=$(echo $stats | cut -d',' -f1)
                    success_std=$(echo $stats | cut -d',' -f2)
                    reward_mean=$(echo $stats | cut -d',' -f3)
                    reward_std=$(echo $stats | cut -d',' -f4)
                    
                    # Save aggregated results to main CSV
                    echo "$method,$morph,$task,$percent,$scale,$success_mean,$success_std,$reward_mean,$reward_std" >> $AGGREGATED_RESULTS
                    
                    echo "  ✅ $morph-$task-${percent}%: Success=${success_mean}±${success_std}%, Reward=${reward_mean}±${reward_std} (${#success_rates[@]} seeds)"
                else
                    echo "  ❌ $morph-$task-${percent}%: No valid results"
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
echo "🎉 PARALLEL MULTI-METHOD EVALUATION COMPLETE!"
echo "🔒 Run ID: $RUN_ID"
echo "⚡ Execution time saved with $MAX_PARALLEL_JOBS parallel jobs"
echo "📊 Aggregated results saved to: $AGGREGATED_RESULTS"
echo "📁 Individual job logs in: $TEMP_JOBS_DIR/"
echo "📋 Parallel execution log: $RESULTS_DIR/parallel_execution_log.txt"
echo "🎥 Videos saved to: $VIDEO_DIR"
echo "======================================================"

# Display the final aggregated results
echo -e "\n📋 Final Aggregated Results:"
echo "============================================="
cat $AGGREGATED_RESULTS
# Debug: Check if aggregated file exists and has content
if [[ -f "$AGGREGATED_RESULTS" ]]; then
    echo "📊 Aggregated results file found with $(wc -l < "$AGGREGATED_RESULTS") lines"
    echo "First few lines of results:"
    head -5 "$AGGREGATED_RESULTS"
else
    echo "❌ Aggregated results file not found: $AGGREGATED_RESULTS"
fi
##################################
         # GPT
###################################
# Export AGGREGATED_RESULTS/RESULTS_DIR/RUN_ID so the Python block can see them
export AGGREGATED_RESULTS="$AGGREGATED_RESULTS"
export RESULTS_DIR="$RESULTS_DIR"
export RUN_ID="$RUN_ID" 
# Generate summary plots with unique filenames (same as before but faster aggregation)
echo -e "\n📈 Generating summary plots..."
python3 << 'EOF'
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sns
import os
import sys
print(f"Python can access environment: AGGREGATED_RESULTS exists = {'AGGREGATED_RESULTS' in os.environ}")
# try:
#     # Read the aggregated results
#     aggregated_file = os.environ.get('AGGREGATED_RESULTS', '')
#     if not aggregated_file or not os.path.exists(aggregated_file):
#         print(f"❌ Cannot find aggregated results file: {aggregated_file}")
#         sys.exit(1)
    
#     # df = pd.read_csv(os.environ['AGGREGATED_RESULTS'])
#     results_dir = os.environ['RESULTS_DIR']
#     run_id = os.environ['RUN_ID']
############ GPT #############################
try:
    # Read the aggregated results
    aggregated_file = os.environ.get('AGGREGATED_RESULTS', '')
    if not aggregated_file or not os.path.exists(aggregated_file):
        print(f"❌ Cannot find aggregated results file: {aggregated_file}")
        sys.exit(1)
    
    # load dataframe from the aggregated CSV
    df = pd.read_csv(aggregated_file)
    results_dir = os.environ.get('RESULTS_DIR')
    run_id = os.environ.get('RUN_ID', 'run_unknown')    
    ################################################################
    # print("\n📊 Aggregated data loaded successfully:")
    # print(f"Shape: {df.shape}")
    # print("\nFirst few rows:")
    # print(df.head(10))
    print("\n📊 Aggregated data loaded successfully:")
    print(f"Shape: {df.shape}")
    print("\nFirst few rows:")
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
    
    plt.suptitle(f'⚡ PARALLEL Multi-Method Analysis (Run: {run_id})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/multi_method_analysis_{run_id}.png', dpi=150, bbox_inches='tight')
    print(f"✅ Main analysis plot saved to '{results_dir}/multi_method_analysis_{run_id}.png'")
    
    # Print summary statistics
    print("\n📊 Summary Statistics:")
    print("="*60)
    for method in methods:
        method_data = df[df['Method'] == method]
        if not method_data.empty:
            baseline_success = method_data[method_data['Percentage_Increase'] == 0]['Success_Rate_Mean'].mean()
            final_success = method_data[method_data['Percentage_Increase'] == method_data['Percentage_Increase'].max()]['Success_Rate_Mean'].mean()
            
            print(f"{method:12} - Baseline: {baseline_success:.1f}%, Final: {final_success:.1f}%, Drop: {baseline_success-final_success:.1f}%")
    
    print("="*60)
    
except Exception as e:
    print(f"❌ Error creating plots: {e}")
    import traceback
    traceback.print_exc()
EOF

# Set environment variables for Python script
# export AGGREGATED_RESULTS RESULTS_DIR RUN_ID
export AGGREGATED_RESULTS="$AGGREGATED_RESULTS"
export RESULTS_DIR="$RESULTS_DIR" 
export RUN_ID="$RUN_ID"

# Clean up temporary job files (optional)
echo -e "\n🧹 Cleaning up temporary job files..."
# Uncomment next line if you want to remove temp files after aggregation
# rm -rf "$TEMP_JOBS_DIR"

echo -e "\n✨ PARALLEL ANALYSIS COMPLETE!"
echo "🔒 Your unique run identifier: $RUN_ID"
echo "⚡ Speedup achieved: ~${MAX_PARALLEL_JOBS}x faster than sequential"
echo "📊 Check results in: $RESULTS_DIR"
echo ""
echo "📋 MULTI-TERMINAL PARALLEL USAGE GUIDE:"
echo "  Terminal 1: Uncomment allnodes methods and run this script"
echo "  Terminal 2: Uncomment avg_nodes methods (current) and run this script" 
echo "  Terminal 3: Uncomment object_only methods and run this script"
echo "  Each terminal runs ${MAX_PARALLEL_JOBS} jobs in parallel automatically!"
echo ""

# cd ModuMorph && git checkout -- . && git pull origin spo && sudo apt-get install bc && sudo apt-get install ffmpeg -y && code down_wandb.py tools/object_generalization_auto.sh

"""
# Force fix the repository issue
sudo apt-get update --allow-releaseinfo-change
sudo apt-get install parallel -y

# Verify installation
which parallel
parallel --version
"""