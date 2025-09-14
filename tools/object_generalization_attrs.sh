# #!/bin/bash

# ##################################################################
# ################ PHYSICAL ATTRIBUTES ROBUSTNESS #################
# ##################################################################

# # --- 1. CONFIGURATION ---
# # Define all artifact paths with their seeds
# declare -A ARTIFACT_PATHS=(
#     # ["allnodes_seed1"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_ALLNODES4_1409-run:v9"
#     # ["allnodes_seed2"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_ALLNODES4_3296-run:v9"
#     # ["allnodes_seed3"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_ALLNODES4_2008-run:v9"
#     ["avg_nodes_seed1"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_avg_nodes_1409-run:v19"
#     ["avg_nodes_seed2"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_avg_nodes_3296-run:v18"
#     ["avg_nodes_seed3"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_avg_nodes_2008-run:v18"
# #     ["object_only_seed1"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_object_only_1409-run:v20"
# #     ["object_only_seed2"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_object_only_3296-run:v18"
# #     ["object_only_seed3"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_object_only4_2008-run:v9"
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

# # Number of episodes to run for each sample
# EPISODES=5

# # Directory to save videos (optional, can be left empty)
# VIDEO_DIR="./analysis_results/physical_attrs_videos/"

# # GLOBAL SAMPLING SEED - CRITICAL: This ensures all computers sample identical values
# SAMPLING_SEED=42

# # Sampling configuration
# NORMAL_SAMPLES_PER_ATTR=10
# EXTREME_SAMPLES_PER_ATTR=2

# # --- END OF CONFIGURATION ---

# # --- 2. SCRIPT LOGIC ---

# # Create results directory
# RESULTS_DIR="./physical_attrs_results"
# mkdir -p $RESULTS_DIR

# # Create main aggregated results file
# AGGREGATED_RESULTS="$RESULTS_DIR/physical_attrs_aggregated.csv"
# echo "Method,Morph,Task,Attribute_Type,Sample_Index,Attribute_Value,Success_Rate_Mean,Success_Rate_Std,Avg_Reward_Mean,Avg_Reward_Std" > $AGGREGATED_RESULTS

# echo "======================================================"
# echo "PHYSICAL ATTRIBUTES ROBUSTNESS EVALUATION"
# echo "Sampling Seed: $SAMPLING_SEED (ensures reproducibility across computers)"
# echo "Methods: Auto-detected from artifact paths"
# echo "Morphs: ${MORPHS[*]}"
# echo "Tasks: ${TASKS[*]}"
# echo "Attributes: Scale, Density, Friction"
# echo "Samples per attribute: ${NORMAL_SAMPLES_PER_ATTR} normal + ${EXTREME_SAMPLES_PER_ATTR} extreme"
# echo "Results will be saved to: $AGGREGATED_RESULTS"
# echo "======================================================"

# # Generate deterministic samples using Python (same across all computers)
# echo "🎲 Generating deterministic samples with seed $SAMPLING_SEED..."
# python3 << EOF
# import numpy as np
# import csv

# # Set the global seed for reproducibility
# np.random.seed($SAMPLING_SEED)

# # Generate normal range samples
# scale_normal = np.random.uniform(0.7, 1.5, $NORMAL_SAMPLES_PER_ATTR)
# density_normal = np.random.uniform(300, 1200, $NORMAL_SAMPLES_PER_ATTR)  # kg/m³
# friction_normal = np.random.uniform(0.3, 1.8, $NORMAL_SAMPLES_PER_ATTR)

# # Add extreme cases
# scale_extreme = np.array([0.3, 2.5])
# density_extreme = np.array([100, 2000])
# friction_extreme = np.array([0.05, 3.0])

# # Combine normal and extreme
# scale_samples = np.concatenate([scale_normal, scale_extreme])
# density_samples = np.concatenate([density_normal, density_extreme])
# friction_samples = np.concatenate([friction_normal, friction_extreme])

# # Save samples to files for bash to read
# with open('$RESULTS_DIR/scale_samples.txt', 'w') as f:
#     for i, val in enumerate(scale_samples):
#         f.write(f"{i},{val:.4f}\n")

# with open('$RESULTS_DIR/density_samples.txt', 'w') as f:
#     for i, val in enumerate(density_samples):
#         f.write(f"{i},{val:.2f}\n")

# with open('$RESULTS_DIR/friction_samples.txt', 'w') as f:
#     for i, val in enumerate(friction_samples):
#         f.write(f"{i},{val:.4f}\n")

# print("📊 Generated samples:")
# print(f"Scale: {scale_samples}")
# print(f"Density: {density_samples}")
# print(f"Friction: {friction_samples}")
# EOF

# # Read the generated samples
# declare -A SCALE_SAMPLES
# declare -A DENSITY_SAMPLES
# declare -A FRICTION_SAMPLES

# while IFS=, read -r idx val; do
#     SCALE_SAMPLES[$idx]=$val
# done < "$RESULTS_DIR/scale_samples.txt"

# while IFS=, read -r idx val; do
#     DENSITY_SAMPLES[$idx]=$val
# done < "$RESULTS_DIR/density_samples.txt"

# while IFS=, read -r idx val; do
#     FRICTION_SAMPLES[$idx]=$val
# done < "$RESULTS_DIR/friction_samples.txt"

# echo "✅ Loaded $(( ${#SCALE_SAMPLES[@]} )) scale samples, $(( ${#DENSITY_SAMPLES[@]} )) density samples, $(( ${#FRICTION_SAMPLES[@]} )) friction samples"

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

# # Automatically detect method names and their seeds
# declare -A METHODS
# declare -A METHOD_SEEDS

# echo "🔍 Auto-detecting methods and seeds from artifact paths..."

# # First pass: identify all methods and their seeds
# for key in "${!ARTIFACT_PATHS[@]}"; do
#     # Try different patterns to extract method name
#     method=""
    
#     # Pattern 1: method_seed[0-9]+ (e.g., "allnodes_seed1")
#     if [[ $key =~ ^(.+)_seed[0-9]+$ ]]; then
#         method="${BASH_REMATCH[1]}"
#     # Pattern 2: method_run_[0-9]+ (e.g., "approach_run_42") 
#     elif [[ $key =~ ^(.+)_run_[0-9]+$ ]]; then
#         method="${BASH_REMATCH[1]}"
#     # Pattern 3: method[0-9]+ (e.g., "baseline1", "baseline2")
#     elif [[ $key =~ ^(.+[^0-9])[0-9]+$ ]]; then
#         method="${BASH_REMATCH[1]}"
#     # Pattern 4: fallback - everything before last underscore
#     elif [[ $key =~ ^(.+)_[^_]+$ ]]; then
#         method="${BASH_REMATCH[1]}"
#     else
#         # No pattern matched, treat whole key as method name
#         method="$key"
#     fi
    
#     # Store method and add this key to its seed list
#     METHODS[$method]=1
#     if [[ -z "${METHOD_SEEDS[$method]}" ]]; then
#         METHOD_SEEDS[$method]="$key"
#     else
#         METHOD_SEEDS[$method]="${METHOD_SEEDS[$method]} $key"
#     fi
# done

# # Display detected structure
# echo "📋 Detected methods and their seeds:"
# for method in "${!METHODS[@]}"; do
#     seed_keys=(${METHOD_SEEDS[$method]})
#     echo "  $method: ${#seed_keys[@]} seeds (${seed_keys[*]})"
# done

# # Define attribute configurations
# declare -A ATTR_CONFIGS
# ATTR_CONFIGS["scale"]="--scale"
# ATTR_CONFIGS["density"]="--density" 
# ATTR_CONFIGS["friction"]="--friction"

# total_combinations=$((${#METHODS[@]} * ${#MORPHS[@]} * ${#TASKS[@]} * 3 * (${NORMAL_SAMPLES_PER_ATTR} + ${EXTREME_SAMPLES_PER_ATTR})))
# current_combo=0

# # Loop through each attribute type
# for attr_type in "scale" "density" "friction"; do
#     echo -e "\n🧪 Testing attribute: $attr_type"
    
#     # Get the appropriate samples array
#     declare -n current_samples="${attr_type^^}_SAMPLES"
#     attr_flag="${ATTR_CONFIGS[$attr_type]}"
    
#     # Loop through each sample for this attribute
#     for sample_idx in "${!current_samples[@]}"; do
#         sample_value="${current_samples[$sample_idx]}"
#         echo -e "\n  📏 Sample ${sample_idx}: ${attr_type}=${sample_value}"
        
#         # Loop through each method
#         for method in "${!METHODS[@]}"; do
#             echo -e "\n    🔬 Method: $method"
            
#             # Get all seed keys for this method
#             seed_keys=(${METHOD_SEEDS[$method]})
            
#             # Loop through each morphology
#             for morph in "${MORPHS[@]}"; do
#                 echo -e "\n      🤖 Morph: $morph"
                
#                 # Loop through each task
#                 for task in "${TASKS[@]}"; do
#                     current_combo=$((current_combo + 1))
#                     echo -e "\n        🎯 Task: $task - Progress: ${current_combo}/${total_combinations}"
                    
#                     # Collect results from all seeds for this method
#                     declare -a success_rates=()
#                     declare -a avg_rewards=()
                    
#                     # Loop through all detected seeds for current method
#                     seed_counter=1
#                     for key in ${seed_keys[@]}; do
#                         run_dir="${ARTIFACT_PATHS[$key]}"
                        
#                         if [[ -z "$run_dir" ]]; then
#                             echo "          ⚠️  Warning: No artifact path for ${key}, skipping..."
#                             continue
#                         fi
                        
#                         echo "          🔄 Running seed ${seed_counter}/${#seed_keys[@]}: ${key}"
                        
#                         # Extract seed from run_dir for reproducibility
#                         seed=$(echo $run_dir | grep -oP '[0-9]+(?=-run)' | tail -1)
#                         seed=${seed:-$seed_counter}
                        
#                         # Construct the command with the specific attribute
#                         if [[ "$attr_type" == "scale" ]]; then
#                             CMD="python tools/obj_geom_seeded.py \
#                               --run_dir $run_dir \
#                               --checkpoint $CHECKPOINT \
#                               --morph $morph \
#                               --task $task \
#                               --base_task $BASE_TASK \
#                               --controller $CONTROLLER \
#                               --episodes $EPISODES \
#                               --save_video $VIDEO_DIR \
#                               --scale $sample_value \
#                               --seed $seed"
#                         elif [[ "$attr_type" == "density" ]]; then
#                             CMD="python tools/obj_geom_seeded.py \
#                               --run_dir $run_dir \
#                               --checkpoint $CHECKPOINT \
#                               --morph $morph \
#                               --task $task \
#                               --base_task $BASE_TASK \
#                               --controller $CONTROLLER \
#                               --episodes $EPISODES \
#                               --save_video $VIDEO_DIR \
#                               --density $sample_value \
#                               --seed $seed"
#                         elif [[ "$attr_type" == "friction" ]]; then
#                             # Friction needs 3 values, use sample_value for all components
#                             CMD="python tools/obj_geom_seeded.py \
#                               --run_dir $run_dir \
#                               --checkpoint $CHECKPOINT \
#                               --morph $morph \
#                               --task $task \
#                               --base_task $BASE_TASK \
#                               --controller $CONTROLLER \
#                               --episodes $EPISODES \
#                               --save_video $VIDEO_DIR \
#                               --friction $sample_value $sample_value $sample_value \
#                               --seed $seed"
#                         fi

#                         # Execute the command and capture output
#                         output=$($CMD 2>&1)
                        
#                         # Extract results from the output
#                         values=$(extract_values "$output")
#                         success_rate=$(echo $values | cut -d',' -f1)
#                         avg_reward=$(echo $values | cut -d',' -f2)
                        
#                         # Store results
#                         success_rates+=($success_rate)
#                         avg_rewards+=($avg_reward)
                        
#                         echo "          ✅ Seed ${seed_counter}: Success=${success_rate}%, Reward=${avg_reward}"
                        
#                         seed_counter=$((seed_counter + 1))
#                     done
                    
#                     # Calculate aggregated statistics
#                     if [ ${#success_rates[@]} -gt 0 ]; then
#                         # Convert arrays to comma-separated strings for Python
#                         success_str=$(IFS=','; echo "${success_rates[*]}")
#                         reward_str=$(IFS=','; echo "${avg_rewards[*]}")
                        
#                         # Calculate mean and std for success rates
#                         success_mean=$(python3 -c "
# import numpy as np
# data = [$success_str]
# print(f'{np.mean(data):.2f}')
# ")
#                         success_std=$(python3 -c "
# import numpy as np
# data = [$success_str]
# print(f'{np.std(data):.2f}')
# ")
                        
#                         # Calculate mean and std for avg rewards  
#                         reward_mean=$(python3 -c "
# import numpy as np
# data = [$reward_str]
# print(f'{np.mean(data):.2f}')
# ")
#                         reward_std=$(python3 -c "
# import numpy as np
# data = [$reward_str]
# print(f'{np.std(data):.2f}')
# ")
                        
#                         # Save aggregated results to main CSV
#                         echo "$method,$morph,$task,$attr_type,$sample_idx,$sample_value,$success_mean,$success_std,$reward_mean,$reward_std" >> $AGGREGATED_RESULTS
                        
#                         echo "        📈 Aggregated - Success: ${success_mean}±${success_std}%, Reward: ${reward_mean}±${reward_std}"
                        
#                         # Save individual seed results for this combination
#                         individual_file="$RESULTS_DIR/individual_${method}_${morph}_${task}_${attr_type}_${sample_idx}.csv"
#                         echo "Seed,Key,Success_Rate,Avg_Reward" > $individual_file
#                         seed_idx=0
#                         for key in ${seed_keys[@]}; do
#                             if [ $seed_idx -lt ${#success_rates[@]} ]; then
#                                 echo "$((seed_idx+1)),$key,${success_rates[$seed_idx]},${avg_rewards[$seed_idx]}" >> $individual_file
#                             fi
#                             seed_idx=$((seed_idx + 1))
#                         done
#                     else
#                         echo "        ❌ No valid results for this combination"
#                         echo "$method,$morph,$task,$attr_type,$sample_idx,$sample_value,0.0,0.0,0.0,0.0" >> $AGGREGATED_RESULTS
#                     fi
                    
#                     # Clear arrays for next iteration
#                     unset success_rates
#                     unset avg_rewards
#                     declare -a success_rates=()
#                     declare -a avg_rewards=()
#                 done
#             done
#         done
#     done
# done

# echo -e "\n======================================================"
# echo "🎉 PHYSICAL ATTRIBUTES EVALUATION COMPLETE!"
# echo "📊 Aggregated results saved to: $AGGREGATED_RESULTS"
# echo "📁 Individual seed results saved to: $RESULTS_DIR/individual_*.csv"
# echo "🎲 Sample files saved to: $RESULTS_DIR/*_samples.txt"
# echo "======================================================"

# # Display the final aggregated results preview
# echo -e "\n📋 Final Aggregated Results Preview (first 20 rows):"
# echo "============================================="
# head -21 $AGGREGATED_RESULTS

# # Generate histogram analysis plots
# echo -e "\n📈 Generating physical attributes analysis plots..."
# python3 << EOF
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import seaborn as sns

# try:
#     # Read the aggregated results
#     df = pd.read_csv("$AGGREGATED_RESULTS")
    
#     print("\\n📊 Physical attributes data loaded successfully:")
#     print(f"Shape: {df.shape}")
#     print(f"Attributes tested: {df['Attribute_Type'].unique()}")
#     print(f"Methods: {df['Method'].unique()}")
#     print("\\nFirst few rows:")
#     print(df.head())
    
#     # Set up the plotting style
#     plt.style.use('default')
#     sns.set_palette("husl")
    
#     # Create comprehensive analysis
#     attributes = df['Attribute_Type'].unique()
#     methods = df['Method'].unique()
    
#     # 1. Histogram-style analysis: Average impact of each attribute
#     fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
#     # Success rate impact by attribute type
#     attr_impact_success = df.groupby(['Attribute_Type', 'Method'])['Success_Rate_Mean'].mean().unstack()
#     attr_impact_success.plot(kind='bar', ax=axes[0,0], rot=45)
#     axes[0,0].set_title('Average Success Rate Impact by Physical Attribute')
#     axes[0,0].set_ylabel('Success Rate (%)')
#     axes[0,0].legend(title='Method')
    
#     # Reward impact by attribute type
#     attr_impact_reward = df.groupby(['Attribute_Type', 'Method'])['Avg_Reward_Mean'].mean().unstack()
#     attr_impact_reward.plot(kind='bar', ax=axes[0,1], rot=45)
#     axes[0,1].set_title('Average Reward Impact by Physical Attribute')
#     axes[0,1].set_ylabel('Average Reward')
#     axes[0,1].legend(title='Method')
    
#     # Box plots showing distribution of performance across attribute samples
#     success_data = []
#     reward_data = []
#     attr_labels = []
#     method_labels = []
    
#     for attr in attributes:
#         for method in methods:
#             attr_method_data = df[(df['Attribute_Type'] == attr) & (df['Method'] == method)]
#             if not attr_method_data.empty:
#                 success_data.extend(attr_method_data['Success_Rate_Mean'].tolist())
#                 reward_data.extend(attr_method_data['Avg_Reward_Mean'].tolist())
#                 attr_labels.extend([attr] * len(attr_method_data))
#                 method_labels.extend([method] * len(attr_method_data))
    
#     box_df = pd.DataFrame({
#         'Success_Rate': success_data,
#         'Reward': reward_data,
#         'Attribute': attr_labels,
#         'Method': method_labels
#     })
    
#     sns.boxplot(data=box_df, x='Attribute', y='Success_Rate', hue='Method', ax=axes[1,0])
#     axes[1,0].set_title('Success Rate Distribution by Attribute Type')
#     axes[1,0].set_ylabel('Success Rate (%)')
    
#     sns.boxplot(data=box_df, x='Attribute', y='Reward', hue='Method', ax=axes[1,1])
#     axes[1,1].set_title('Reward Distribution by Attribute Type')
#     axes[1,1].set_ylabel('Average Reward')
    
#     plt.suptitle('Physical Attributes Robustness Analysis', fontsize=16, fontweight='bold')
#     plt.tight_layout()
#     plt.savefig('$RESULTS_DIR/physical_attrs_analysis.png', dpi=150, bbox_inches='tight')
#     print("✅ Main analysis plot saved to '$RESULTS_DIR/physical_attrs_analysis.png'")
    
#     # 2. Individual attribute sensitivity plots
#     fig, axes = plt.subplots(len(attributes), len(methods), figsize=(5*len(methods), 4*len(attributes)), sharey=True)
#     if len(attributes) == 1:
#         axes = axes.reshape(1, -1)
#     if len(methods) == 1:
#         axes = axes.reshape(-1, 1)
    
#     for i, attr in enumerate(attributes):
#         for j, method in enumerate(methods):
#             subset = df[(df['Attribute_Type'] == attr) & (df['Method'] == method)]
#             if not subset.empty:
#                 # Sort by attribute value for better visualization
#                 subset = subset.sort_values('Attribute_Value')
#                 axes[i,j].scatter(subset['Attribute_Value'], subset['Success_Rate_Mean'], 
#                                 alpha=0.7, s=50, c=subset['Avg_Reward_Mean'], cmap='viridis')
#                 axes[i,j].set_title(f'{method} - {attr}')
#                 axes[i,j].set_xlabel(f'{attr} Value')
#                 if j == 0:
#                     axes[i,j].set_ylabel('Success Rate (%)')
#                 axes[i,j].grid(True, alpha=0.3)
    
#     plt.suptitle('Attribute Sensitivity Analysis (Color = Reward)', fontsize=16, fontweight='bold')
#     plt.tight_layout()
#     plt.savefig('$RESULTS_DIR/attribute_sensitivity.png', dpi=150, bbox_inches='tight')
#     print("✅ Sensitivity analysis plot saved to '$RESULTS_DIR/attribute_sensitivity.png'")
    
#     # 3. Statistical summary
#     print("\\n📊 Statistical Summary:")
#     print("="*80)
    
#     # Calculate variance/sensitivity for each attribute-method combination
#     sensitivity_results = []
#     for attr in attributes:
#         for method in methods:
#             subset = df[(df['Attribute_Type'] == attr) & (df['Method'] == method)]
#             if len(subset) > 1:
#                 success_var = subset['Success_Rate_Mean'].var()
#                 reward_var = subset['Avg_Reward_Mean'].var()
#                 success_range = subset['Success_Rate_Mean'].max() - subset['Success_Rate_Mean'].min()
#                 reward_range = subset['Avg_Reward_Mean'].max() - subset['Avg_Reward_Mean'].min()
                
#                 sensitivity_results.append({
#                     'Attribute': attr,
#                     'Method': method,
#                     'Success_Variance': success_var,
#                     'Reward_Variance': reward_var,
#                     'Success_Range': success_range,
#                     'Reward_Range': reward_range
#                 })
    
#     sens_df = pd.DataFrame(sensitivity_results)
#     if not sens_df.empty:
#         print("🎯 ATTRIBUTE SENSITIVITY RANKING (by success rate variance):")
#         print("-" * 60)
#         for method in methods:
#             method_data = sens_df[sens_df['Method'] == method].sort_values('Success_Variance', ascending=False)
#             print(f"\\n{method}:")
#             for _, row in method_data.iterrows():
#                 print(f"  {row['Attribute']:8} - Variance: {row['Success_Variance']:8.2f}, Range: {row['Success_Range']:6.1f}%")
    
#     print("\\n" + "="*80)
    
#     # Save sensitivity summary
#     if not sens_df.empty:
#         sens_df.to_csv('$RESULTS_DIR/sensitivity_summary.csv', index=False)
#         print("✅ Sensitivity summary saved to '$RESULTS_DIR/sensitivity_summary.csv'")
    
# except Exception as e:
#     print(f"❌ Error creating plots: {e}")
#     import traceback
#     traceback.print_exc()

# EOF

# echo -e "\n✨ Physical attributes analysis complete! Check the results directory: $RESULTS_DIR"
# echo "🎲 Remember: All samples were generated with seed $SAMPLING_SEED for reproducibility across computers"



#!/bin/bash

##################################################################
################ PHYSICAL ATTRIBUTES ROBUSTNESS #################
##################################################################

# --- 1. CONFIGURATION ---
# Define all artifact paths with their seeds
declare -A ARTIFACT_PATHS=(
    # ["allnodes_seed1"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_ALLNODES4_1409-run:v9"
    # ["allnodes_seed2"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_ALLNODES4_3296-run:v9"
    # ["allnodes_seed3"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_ALLNODES4_2008-run:v9"
    ["avg_nodes_seed1"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_avg_nodes_1409-run:v19"
    ["avg_nodes_seed2"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_avg_nodes_3296-run:v18"
    ["avg_nodes_seed3"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_avg_nodes_2008-run:v18"
#     ["object_only_seed1"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_object_only_1409-run:v20"
#     ["object_only_seed2"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_object_only_3296-run:v18"
#     ["object_only_seed3"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_object_only4_2008-run:v9"
)

# Define checkpoint file (assuming same for all)
CHECKPOINT="checkpoint_600.pt"

# Define all morphologies to test
MORPHS=("Jaco" "Kinova3") #"IIWA" "Sawyer")

# Define all tasks to test  
TASKS=("LiftBall" "LiftCylinder")

# The BASE task from which the learned skill/embedding should be drawn
BASE_TASK="Lift"

# Controller used for this robot-task pair
CONTROLLER="OSC_POSE"

# Number of episodes to run for each sample
EPISODES=1

# Directory to save videos (optional, can be left empty)
VIDEO_DIR="./analysis_results/physical_attrs_videos/"

# GLOBAL SAMPLING SEED - CRITICAL: This ensures all computers sample identical values
SAMPLING_SEED=42

# Sampling configuration
NORMAL_SAMPLES_PER_ATTR=5
EXTREME_SAMPLES_PER_ATTR=2

# --- PARALLEL EXECUTION SETTINGS ---
MAX_PARALLEL_JOBS=6        # Conservative for physical simulation load
GPU_JOBS_LIMIT=3          # Fewer GPU jobs for physics-heavy tasks
MEMORY_PER_JOB="2G"       # More memory for physics simulations
RUN_ID="${USER}_$(hostname)_$$_$(date +%s%N | cut -c1-13)"

echo "🔒 Multi-terminal safety enabled. Run ID: $RUN_ID"
echo "⚡ Parallel execution: MAX_JOBS=$MAX_PARALLEL_JOBS, GPU_LIMIT=$GPU_JOBS_LIMIT"

# --- END OF CONFIGURATION ---

# --- 2. SCRIPT LOGIC ---

# Create results directory (unique per run)
RESULTS_DIR="./physical_attrs_results_${RUN_ID}"
mkdir -p $RESULTS_DIR

# Create main aggregated results file
AGGREGATED_RESULTS="$RESULTS_DIR/physical_attrs_aggregated.csv"
echo "Method,Morph,Task,Attribute_Type,Sample_Index,Attribute_Value,Success_Rate_Mean,Success_Rate_Std,Avg_Reward_Mean,Avg_Reward_Std" > $AGGREGATED_RESULTS

echo "======================================================"
echo "PHYSICAL ATTRIBUTES ROBUSTNESS EVALUATION"
echo "Sampling Seed: $SAMPLING_SEED (ensures reproducibility across computers)"
echo "Methods: Auto-detected from artifact paths"
echo "Morphs: ${MORPHS[*]}"
echo "Tasks: ${TASKS[*]}"
echo "Attributes: Scale, Density, Friction"
echo "Samples per attribute: ${NORMAL_SAMPLES_PER_ATTR} normal + ${EXTREME_SAMPLES_PER_ATTR} extreme"
echo "Results will be saved to: $AGGREGATED_RESULTS"
echo "======================================================"

# Generate deterministic samples using Python (same across all computers)
echo "🎲 Generating deterministic samples with seed $SAMPLING_SEED..."
python3 << EOF
import numpy as np
import csv

# Set the global seed for reproducibility
np.random.seed($SAMPLING_SEED)

# Generate normal range samples
scale_normal = np.random.uniform(0.7, 1.5, $NORMAL_SAMPLES_PER_ATTR)
density_normal = np.random.uniform(300, 1200, $NORMAL_SAMPLES_PER_ATTR)  # kg/m³
friction_normal = np.random.uniform(0.3, 1.8, $NORMAL_SAMPLES_PER_ATTR)

# Add extreme cases
scale_extreme = np.array([0.3, 2.5])
density_extreme = np.array([100, 2000])
friction_extreme = np.array([0.05, 3.0])

# Combine normal and extreme
scale_samples = np.concatenate([scale_normal, scale_extreme])
density_samples = np.concatenate([density_normal, density_extreme])
friction_samples = np.concatenate([friction_normal, friction_extreme])

# Save samples to files for bash to read
with open('$RESULTS_DIR/scale_samples.txt', 'w') as f:
    for i, val in enumerate(scale_samples):
        f.write(f"{i},{val:.4f}\n")

with open('$RESULTS_DIR/density_samples.txt', 'w') as f:
    for i, val in enumerate(density_samples):
        f.write(f"{i},{val:.2f}\n")

with open('$RESULTS_DIR/friction_samples.txt', 'w') as f:
    for i, val in enumerate(friction_samples):
        f.write(f"{i},{val:.4f}\n")

print("📊 Generated samples:")
print(f"Scale: {scale_samples}")
print(f"Density: {density_samples}")
print(f"Friction: {friction_samples}")
EOF

# Read the generated samples
declare -A SCALE_SAMPLES
declare -A DENSITY_SAMPLES
declare -A FRICTION_SAMPLES

while IFS=, read -r idx val; do
    SCALE_SAMPLES[$idx]=$val
done < "$RESULTS_DIR/scale_samples.txt"

while IFS=, read -r idx val; do
    DENSITY_SAMPLES[$idx]=$val
done < "$RESULTS_DIR/density_samples.txt"

while IFS=, read -r idx val; do
    FRICTION_SAMPLES[$idx]=$val
done < "$RESULTS_DIR/friction_samples.txt"

echo "✅ Loaded $(( ${#SCALE_SAMPLES[@]} )) scale samples, $(( ${#DENSITY_SAMPLES[@]} )) density samples, $(( ${#FRICTION_SAMPLES[@]} )) friction samples"

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

# Automatically detect method names and their seeds
declare -A METHODS
declare -A METHOD_SEEDS

echo "🔍 Auto-detecting methods and seeds from artifact paths..."

# First pass: identify all methods and their seeds
for key in "${!ARTIFACT_PATHS[@]}"; do
    # Try different patterns to extract method name
    method=""
    
    # Pattern 1: method_seed[0-9]+ (e.g., "allnodes_seed1")
    if [[ $key =~ ^(.+)_seed[0-9]+$ ]]; then
        method="${BASH_REMATCH[1]}"
    # Pattern 2: method_run_[0-9]+ (e.g., "approach_run_42") 
    elif [[ $key =~ ^(.+)_run_[0-9]+$ ]]; then
        method="${BASH_REMATCH[1]}"
    # Pattern 3: method[0-9]+ (e.g., "baseline1", "baseline2")
    elif [[ $key =~ ^(.+[^0-9])[0-9]+$ ]]; then
        method="${BASH_REMATCH[1]}"
    # Pattern 4: fallback - everything before last underscore
    elif [[ $key =~ ^(.+)_[^_]+$ ]]; then
        method="${BASH_REMATCH[1]}"
    else
        # No pattern matched, treat whole key as method name
        method="$key"
    fi
    
    # Store method and add this key to its seed list
    METHODS[$method]=1
    if [[ -z "${METHOD_SEEDS[$method]}" ]]; then
        METHOD_SEEDS[$method]="$key"
    else
        METHOD_SEEDS[$method]="${METHOD_SEEDS[$method]} $key"
    fi
done

# Display detected structure
echo "📋 Detected methods and their seeds:"
for method in "${!METHODS[@]}"; do
    seed_keys=(${METHOD_SEEDS[$method]})
    echo "  $method: ${#seed_keys[@]} seeds (${seed_keys[*]})"
done

# Define attribute configurations
declare -A ATTR_CONFIGS
ATTR_CONFIGS["scale"]="--scale"
ATTR_CONFIGS["density"]="--density" 
ATTR_CONFIGS["friction"]="--friction"

total_combinations=$((${#METHODS[@]} * ${#MORPHS[@]} * ${#TASKS[@]} * 3 * (${NORMAL_SAMPLES_PER_ATTR} + ${EXTREME_SAMPLES_PER_ATTR})))
current_combo=0

# --- PARALLELIZED JOBS CREATION & EXECUTION ---

# Function to run a single evaluation job
run_single_attr_job() {
    local job_id="$1"
    local method="$2" 
    local key="$3"
    local run_dir="$4"
    local morph="$5"
    local task="$6" 
    local attr_type="$7"
    local sample_idx="$8"
    local sample_value="$9"
    local seed="${10}"
    
    local job_output_file="$TEMP_JOBS_DIR/job_${job_id}_${method}_${morph}_${task}_${attr_type}_${sample_idx}.txt"
    
    # Build command based on attribute type
    local CMD="python tools/obj_geom_seeded.py \
      --run_dir $run_dir \
      --checkpoint $CHECKPOINT \
      --morph $morph \
      --task $task \
      --base_task $BASE_TASK \
      --controller $CONTROLLER \
      --episodes $EPISODES \
      --save_video $VIDEO_DIR \
      --seed $seed"
      
    if [[ "$attr_type" == "scale" ]]; then
        CMD="$CMD --scale $sample_value"
    elif [[ "$attr_type" == "density" ]]; then
        CMD="$CMD --density $sample_value"
    elif [[ "$attr_type" == "friction" ]]; then
        CMD="$CMD --friction $sample_value $sample_value $sample_value"
    fi
    
    # Execute and capture results
    local output
    if output=$($CMD 2>&1); then
        local values=$(extract_values "$output")
        local success_rate=$(echo $values | cut -d',' -f1)
        local avg_reward=$(echo $values | cut -d',' -f2)
        echo "SUCCESS,$success_rate,$avg_reward" >> "$job_output_file"
    else
        echo "FAILED,0.0,0.0" >> "$job_output_file"
        echo "Error: $output" >> "$job_output_file"
    fi
}

export -f run_single_attr_job
export -f extract_values

# Create temporary directory for jobs
TEMP_JOBS_DIR="$RESULTS_DIR/temp_jobs"
mkdir -p $TEMP_JOBS_DIR

# Generate job list
JOB_LIST_FILE="$TEMP_JOBS_DIR/job_list.txt"
> $JOB_LIST_FILE
job_counter=0

# Generate all job combinations
for attr_type in "scale" "density" "friction"; do
    declare -n current_samples="${attr_type^^}_SAMPLES"
    
    for sample_idx in "${!current_samples[@]}"; do
        sample_value="${current_samples[$sample_idx]}"
        
        for method in "${!METHODS[@]}"; do
            seed_keys=(${METHOD_SEEDS[$method]})
            
            for morph in "${MORPHS[@]}"; do
                for task in "${TASKS[@]}"; do
                    for key in ${seed_keys[@]}; do
                        run_dir="${ARTIFACT_PATHS[$key]}"
                        seed=$(echo $run_dir | grep -oP '[0-9]+(?=-run)' | tail -1)
                        seed=${seed:-1}
                        
                        job_counter=$((job_counter + 1))
                        echo "$job_counter $method $key $run_dir $morph $task $attr_type $sample_idx $sample_value $seed" >> $JOB_LIST_FILE
                    done
                done
            done
        done
    done
done

echo "📊 Generated $job_counter parallel jobs"
echo "⚡ Starting parallel execution..."

# Export runtime variables so GNU parallel workers can see them
export TEMP_JOBS_DIR
export RESULTS_DIR
export AGGREGATED_RESULTS
export CHECKPOINT
export BASE_TASK
export CONTROLLER
export EPISODES
export VIDEO_DIR
export RUN_ID

parallel \
    --jobs $MAX_PARALLEL_JOBS \
    --colsep ' ' \
    --progress \
    --bar \
    --eta \
    --joblog "$RESULTS_DIR/parallel_execution_log.txt" \
    --memfree $MEMORY_PER_JOB \
    run_single_attr_job {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} :::: $JOB_LIST_FILE

echo "✅ All parallel jobs completed!"

# Aggregate results from parallel jobs
echo "📊 Aggregating results from parallel jobs..."

for attr_type in "scale" "density" "friction"; do
    declare -n current_samples="${attr_type^^}_SAMPLES"
    
    for sample_idx in "${!current_samples[@]}"; do
        sample_value="${current_samples[$sample_idx]}"
        
        for method in "${!METHODS[@]}"; do
            seed_keys=(${METHOD_SEEDS[$method]})
            
            for morph in "${MORPHS[@]}"; do
                for task in "${TASKS[@]}"; do
                    # Collect results from all seeds
                    declare -a success_rates=()
                    declare -a avg_rewards=()
                    
                    for key in ${seed_keys[@]}; do
                        job_file=$(find "$TEMP_JOBS_DIR" -name "job_*_${method}_${morph}_${task}_${attr_type}_${sample_idx}.txt" | head -1)
                        if [[ -f "$job_file" ]]; then
                            result_line=$(grep "^SUCCESS," "$job_file" 2>/dev/null || echo "FAILED,0.0,0.0")
                            success_rate=$(echo $result_line | cut -d',' -f2)
                            avg_reward=$(echo $result_line | cut -d',' -f3)
                            success_rates+=($success_rate)
                            avg_rewards+=($avg_reward)
                        fi
                    done
                    
                    # Calculate statistics and save to aggregated results
                    if [ ${#success_rates[@]} -gt 0 ]; then
                        success_str=$(IFS=','; echo "${success_rates[*]}")
                        reward_str=$(IFS=','; echo "${avg_rewards[*]}")
                        
                        success_mean=$(python3 -c "import numpy as np; data = [$success_str]; print(f'{np.mean(data):.2f}')")
                        success_std=$(python3 -c "import numpy as np; data = [$success_str]; print(f'{np.std(data):.2f}')")
                        reward_mean=$(python3 -c "import numpy as np; data = [$reward_str]; print(f'{np.mean(data):.2f}')")
                        reward_std=$(python3 -c "import numpy as np; data = [$reward_str]; print(f'{np.std(data):.2f}')")
                        
                        echo "$method,$morph,$task,$attr_type,$sample_idx,$sample_value,$success_mean,$success_std,$reward_mean,$reward_std" >> $AGGREGATED_RESULTS
                    fi
                    
                    unset success_rates avg_rewards
                    declare -a success_rates=() avg_rewards=()
                done
            done
        done
    done
done

echo -e "\n======================================================"
echo "🎉 PHYSICAL ATTRIBUTES EVALUATION COMPLETE!"
echo "📊 Aggregated results saved to: $AGGREGATED_RESULTS"
echo "📁 Individual seed results saved to: $RESULTS_DIR/individual_*.csv"
echo "🎲 Sample files saved to: $RESULTS_DIR/*_samples.txt"
echo "======================================================"

# Display the final aggregated results preview
echo -e "\n📋 Final Aggregated Results Preview (first 20 rows):"
echo "============================================="
# head -21 $AGGREGATED_RESULTS
export AGGREGATED_RESULTS="$AGGREGATED_RESULTS"
export RESULTS_DIR="$RESULTS_DIR"
export RUN_ID="$RUN_ID" 
# Generate histogram analysis plots
echo -e "\n📈 Generating physical attributes analysis plots..."
python3 << EOF
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

try:
    # Read the aggregated results
    df = pd.read_csv("$AGGREGATED_RESULTS")
    
    print("\\n📊 Physical attributes data loaded successfully:")
    print(f"Shape: {df.shape}")
    print(f"Attributes tested: {df['Attribute_Type'].unique()}")
    print(f"Methods: {df['Method'].unique()}")
    print("\\nFirst few rows:")
    print(df.head())
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create comprehensive analysis
    attributes = df['Attribute_Type'].unique()
    methods = df['Method'].unique()
    
    # 1. Histogram-style analysis: Average impact of each attribute
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Success rate impact by attribute type
    attr_impact_success = df.groupby(['Attribute_Type', 'Method'])['Success_Rate_Mean'].mean().unstack()
    attr_impact_success.plot(kind='bar', ax=axes[0,0], rot=45)
    axes[0,0].set_title('Average Success Rate Impact by Physical Attribute')
    axes[0,0].set_ylabel('Success Rate (%)')
    axes[0,0].legend(title='Method')
    
    # Reward impact by attribute type
    attr_impact_reward = df.groupby(['Attribute_Type', 'Method'])['Avg_Reward_Mean'].mean().unstack()
    attr_impact_reward.plot(kind='bar', ax=axes[0,1], rot=45)
    axes[0,1].set_title('Average Reward Impact by Physical Attribute')
    axes[0,1].set_ylabel('Average Reward')
    axes[0,1].legend(title='Method')
    
    # Box plots showing distribution of performance across attribute samples
    success_data = []
    reward_data = []
    attr_labels = []
    method_labels = []
    
    for attr in attributes:
        for method in methods:
            attr_method_data = df[(df['Attribute_Type'] == attr) & (df['Method'] == method)]
            if not attr_method_data.empty:
                success_data.extend(attr_method_data['Success_Rate_Mean'].tolist())
                reward_data.extend(attr_method_data['Avg_Reward_Mean'].tolist())
                attr_labels.extend([attr] * len(attr_method_data))
                method_labels.extend([method] * len(attr_method_data))
    
    box_df = pd.DataFrame({
        'Success_Rate': success_data,
        'Reward': reward_data,
        'Attribute': attr_labels,
        'Method': method_labels
    })
    
    sns.boxplot(data=box_df, x='Attribute', y='Success_Rate', hue='Method', ax=axes[1,0])
    axes[1,0].set_title('Success Rate Distribution by Attribute Type')
    axes[1,0].set_ylabel('Success Rate (%)')
    
    sns.boxplot(data=box_df, x='Attribute', y='Reward', hue='Method', ax=axes[1,1])
    axes[1,1].set_title('Reward Distribution by Attribute Type')
    axes[1,1].set_ylabel('Average Reward')
    
    plt.suptitle('Physical Attributes Robustness Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('$RESULTS_DIR/physical_attrs_analysis.png', dpi=150, bbox_inches='tight')
    print("✅ Main analysis plot saved to '$RESULTS_DIR/physical_attrs_analysis.png'")
    
    # 2. Individual attribute sensitivity plots
    fig, axes = plt.subplots(len(attributes), len(methods), figsize=(5*len(methods), 4*len(attributes)), sharey=True)
    if len(attributes) == 1:
        axes = axes.reshape(1, -1)
    if len(methods) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, attr in enumerate(attributes):
        for j, method in enumerate(methods):
            subset = df[(df['Attribute_Type'] == attr) & (df['Method'] == method)]
            if not subset.empty:
                # Sort by attribute value for better visualization
                subset = subset.sort_values('Attribute_Value')
                axes[i,j].scatter(subset['Attribute_Value'], subset['Success_Rate_Mean'], 
                                alpha=0.7, s=50, c=subset['Avg_Reward_Mean'], cmap='viridis')
                axes[i,j].set_title(f'{method} - {attr}')
                axes[i,j].set_xlabel(f'{attr} Value')
                if j == 0:
                    axes[i,j].set_ylabel('Success Rate (%)')
                axes[i,j].grid(True, alpha=0.3)
    
    plt.suptitle('Attribute Sensitivity Analysis (Color = Reward)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('$RESULTS_DIR/attribute_sensitivity.png', dpi=150, bbox_inches='tight')
    print("✅ Sensitivity analysis plot saved to '$RESULTS_DIR/attribute_sensitivity.png'")
    
    # 3. Statistical summary
    print("\\n📊 Statistical Summary:")
    print("="*80)
    
    # Calculate variance/sensitivity for each attribute-method combination
    sensitivity_results = []
    for attr in attributes:
        for method in methods:
            subset = df[(df['Attribute_Type'] == attr) & (df['Method'] == method)]
            if len(subset) > 1:
                success_var = subset['Success_Rate_Mean'].var()
                reward_var = subset['Avg_Reward_Mean'].var()
                success_range = subset['Success_Rate_Mean'].max() - subset['Success_Rate_Mean'].min()
                reward_range = subset['Avg_Reward_Mean'].max() - subset['Avg_Reward_Mean'].min()
                
                sensitivity_results.append({
                    'Attribute': attr,
                    'Method': method,
                    'Success_Variance': success_var,
                    'Reward_Variance': reward_var,
                    'Success_Range': success_range,
                    'Reward_Range': reward_range
                })
    
    sens_df = pd.DataFrame(sensitivity_results)
    if not sens_df.empty:
        print("🎯 ATTRIBUTE SENSITIVITY RANKING (by success rate variance):")
        print("-" * 60)
        for method in methods:
            method_data = sens_df[sens_df['Method'] == method].sort_values('Success_Variance', ascending=False)
            print(f"\\n{method}:")
            for _, row in method_data.iterrows():
                print(f"  {row['Attribute']:8} - Variance: {row['Success_Variance']:8.2f}, Range: {row['Success_Range']:6.1f}%")
    
    print("\\n" + "="*80)
    
    # Save sensitivity summary
    if not sens_df.empty:
        sens_df.to_csv('$RESULTS_DIR/sensitivity_summary.csv', index=False)
        print("✅ Sensitivity summary saved to '$RESULTS_DIR/sensitivity_summary.csv'")
    
except Exception as e:
    print(f"❌ Error creating plots: {e}")
    import traceback
    traceback.print_exc()

EOF

echo -e "\n✨ Physical attributes analysis complete! Check the results directory: $RESULTS_DIR"
echo "🎲 Remember: All samples were generated with seed $SAMPLING_SEED for reproducibility across computers"
