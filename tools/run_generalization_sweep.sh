# #!/bin/bash

# # Configuration
# RUN_DIR="./artifacts/Robosuite-v0-MR-ST-MR-MT_object_only_JNT_3296-run:v3/"
# CHECKPOINT="checkpoint_300.pt"
# MORPH="Kinova3"  # Change this as needed
# TASK="Door"     # Change this as needed
# CONTROLLER="JOINT_VELOCITY"  # Change this as needed
# EPISODES=5

# # Output file for results
# RESULTS_FILE="size_generalization_results.txt"
# echo "Size_Percent,Success_Rate,Avg_Reward" > $RESULTS_FILE

# echo "========================================="
# echo "Object Size Generalization Evaluation"
# echo "Task: $TASK, Morph: $MORPH"
# echo "========================================="

# # Define the size values based on task
# if [ "$TASK" == "Lift" ]; then
#   # Cube sizes for Lift task (original: 0.02 0.022 0.0215)
#   SIZE_5="0.0210 0.0231 0.0226"
#   SIZE_10="0.0220 0.0242 0.0237"
#   SIZE_15="0.0230 0.0253 0.0247"
#   SIZE_20="0.0240 0.0264 0.0258"
#   SIZE_25="0.0250 0.0275 0.0269"
#   SIZE_30="0.0260 0.0286 0.0280"
#   SIZE_40="0.0280 0.0308 0.0301"
#   SIZE_50="0.0300 0.0330 0.0323"
#   SIZE_60="0.0320 0.0352 0.0344"
#   SIZE_70="0.0340 0.0374 0.0366"
#   SIZE_80="0.0360 0.0396 0.0387"
#   SIZE_90="0.0380 0.0418 0.0409"
#   SIZE_100="0.0400 0.0440 0.0430"
# elif [ "$TASK" == "Door" ]; then
#   # Door handle sizes (original: 0.075 0.015 0.02)
#   SIZE_5="0.0788 0.0158 0.0210"
#   SIZE_10="0.0825 0.0165 0.0220"
#   SIZE_15="0.0863 0.0173 0.0230"
#   SIZE_20="0.0900 0.0180 0.0240"
#   SIZE_25="0.0938 0.0188 0.0250"
#   SIZE_30="0.0975 0.0195 0.0260"
#   SIZE_40="0.1050 0.0210 0.0280"
#   SIZE_50="0.1125 0.0225 0.0300"
#   SIZE_60="0.1200 0.0240 0.0320"
#   SIZE_70="0.1275 0.0255 0.0340"
#   SIZE_80="0.1350 0.0270 0.0360"
#   SIZE_90="0.1425 0.0285 0.0380"
#   SIZE_100="0.1500 0.0300 0.0400"
# else
#   echo "Warning: Unknown task $TASK, using default Lift sizes"
#   SIZE_5="0.0210 0.0231 0.0226"
#   SIZE_10="0.0220 0.0242 0.0237"
#   SIZE_15="0.0230 0.0253 0.0247"
#   SIZE_20="0.0240 0.0264 0.0258"
#   SIZE_25="0.0250 0.0275 0.0269"
#   SIZE_30="0.0260 0.0286 0.0280"
#   SIZE_40="0.0280 0.0308 0.0301"
#   SIZE_50="0.0300 0.0330 0.0323"
#   SIZE_60="0.0320 0.0352 0.0344"
#   SIZE_70="0.0340 0.0374 0.0366"
#   SIZE_80="0.0360 0.0396 0.0387"
#   SIZE_90="0.0380 0.0418 0.0409"
#   SIZE_100="0.0400 0.0440 0.0430"
# fi

# # Function to extract values from output
# extract_values() {
#     local output="$1"
    
#     # Extract success rate - looking for "Success Rate: X.X%"
#     local success_rate=$(echo "$output" | grep -E "Success Rate:" | tail -1 | sed -E 's/.*Success Rate:[[:space:]]*([0-9.]+)%.*/\1/')
    
#     # Extract average reward - looking for "Avg. Reward:  X.XX ± Y.YY"
#     local avg_reward=$(echo "$output" | grep -E "Avg\. Reward:" | tail -1 | awk '{for(i=1;i<=NF;i++) if($i=="Reward:") print $(i+1)}' | sed 's/[^0-9.-]//g')
    
#     # If extraction failed, try alternative patterns
#     if [ -z "$success_rate" ]; then
#         success_rate="0.0"
#         echo "Warning: Could not extract success rate, defaulting to 0.0" >&2
#     fi
    
#     if [ -z "$avg_reward" ]; then
#         avg_reward="0.0"
#         echo "Warning: Could not extract average reward, defaulting to 0.0" >&2
#     fi
    
#     echo "$success_rate,$avg_reward"
# }

# # Test with 0% increase (original size)
# echo -e "\n📊 Testing with 0% size increase (original)..."
# output=$(python tools/obj_geom_eval.py \
#   --run_dir $RUN_DIR \
#   --checkpoint $CHECKPOINT \
#   --morph $MORPH \
#   --task $TASK \
#   --controller $CONTROLLER \
#   --save_video ./test_varObject_vids/ \
#   --episodes $EPISODES 2>&1)

# # Debug: print last few lines of output
# echo "=== Debug output (last 15 lines) ===" >&2
# echo "$output" | tail -15 >&2
# echo "===================================" >&2

# values=$(extract_values "$output")
# success_rate=$(echo $values | cut -d',' -f1)
# avg_reward=$(echo $values | cut -d',' -f2)
# echo "0,$success_rate,$avg_reward" >> $RESULTS_FILE
# echo "✅ 0% - Success Rate: ${success_rate}%, Avg Reward: ${avg_reward}"

# # Test with 5% increase
# echo -e "\n📊 Testing with 5% size increase..."
# output=$(python tools/obj_geom_eval.py \
#   --run_dir $RUN_DIR \
#   --checkpoint $CHECKPOINT \
#   --morph $MORPH \
#   --task $TASK \
#   --controller $CONTROLLER \
#   --episodes $EPISODES \
#   --save_video ./test_varObject_vids/ \
#   --new-geom-size $SIZE_5 2>&1)

# values=$(extract_values "$output")
# success_rate=$(echo $values | cut -d',' -f1)
# avg_reward=$(echo $values | cut -d',' -f2)
# echo "5,$success_rate,$avg_reward" >> $RESULTS_FILE
# echo "✅ 5% - Success Rate: ${success_rate}%, Avg Reward: ${avg_reward}"

# # Test with 10% increase
# echo -e "\n📊 Testing with 10% size increase..."
# output=$(python tools/obj_geom_eval.py \
#   --run_dir $RUN_DIR \
#   --checkpoint $CHECKPOINT \
#   --morph $MORPH \
#   --task $TASK \
#   --controller $CONTROLLER \
#   --episodes $EPISODES \
#   --save_video ./test_varObject_vids/ \
#   --new-geom-size $SIZE_10 2>&1)

# values=$(extract_values "$output")
# success_rate=$(echo $values | cut -d',' -f1)
# avg_reward=$(echo $values | cut -d',' -f2)
# echo "10,$success_rate,$avg_reward" >> $RESULTS_FILE
# echo "✅ 10% - Success Rate: ${success_rate}%, Avg Reward: ${avg_reward}"

# # Test with 15% increase
# echo -e "\n📊 Testing with 15% size increase..."
# output=$(python tools/obj_geom_eval.py \
#   --run_dir $RUN_DIR \
#   --checkpoint $CHECKPOINT \
#   --morph $MORPH \
#   --task $TASK \
#   --controller $CONTROLLER \
#   --episodes $EPISODES \
#   --save_video ./test_varObject_vids/ \
#   --new-geom-size $SIZE_15 2>&1)

# values=$(extract_values "$output")
# success_rate=$(echo $values | cut -d',' -f1)
# avg_reward=$(echo $values | cut -d',' -f2)
# echo "15,$success_rate,$avg_reward" >> $RESULTS_FILE
# echo "✅ 15% - Success Rate: ${success_rate}%, Avg Reward: ${avg_reward}"

# # Test with 20% increase
# echo -e "\n📊 Testing with 20% size increase..."
# output=$(python tools/obj_geom_eval.py \
#   --run_dir $RUN_DIR \
#   --checkpoint $CHECKPOINT \
#   --morph $MORPH \
#   --task $TASK \
#   --controller $CONTROLLER \
#   --episodes $EPISODES \
#   --save_video ./test_varObject_vids/ \
#   --new-geom-size $SIZE_20 2>&1)

# values=$(extract_values "$output")
# success_rate=$(echo $values | cut -d',' -f1)
# avg_reward=$(echo $values | cut -d',' -f2)
# echo "20,$success_rate,$avg_reward" >> $RESULTS_FILE
# echo "✅ 20% - Success Rate: ${success_rate}%, Avg Reward: ${avg_reward}"

# # Test with 25% increase
# echo -e "\n📊 Testing with 25% size increase..."
# output=$(python tools/obj_geom_eval.py \
#   --run_dir $RUN_DIR \
#   --checkpoint $CHECKPOINT \
#   --morph $MORPH \
#   --task $TASK \
#   --controller $CONTROLLER \
#   --episodes $EPISODES \
#   --save_video ./test_varObject_vids/ \
#   --new-geom-size $SIZE_25 2>&1)

# values=$(extract_values "$output")
# success_rate=$(echo $values | cut -d',' -f1)
# avg_reward=$(echo $values | cut -d',' -f2)
# echo "25,$success_rate,$avg_reward" >> $RESULTS_FILE
# echo "✅ 25% - Success Rate: ${success_rate}%, Avg Reward: ${avg_reward}"

# # Test with 30% increase
# echo -e "\n📊 Testing with 30% size increase..."
# output=$(python tools/obj_geom_eval.py \
#   --run_dir $RUN_DIR \
#   --checkpoint $CHECKPOINT \
#   --morph $MORPH \
#   --task $TASK \
#   --controller $CONTROLLER \
#   --episodes $EPISODES \
#   --save_video ./test_varObject_vids/ \
#   --new-geom-size $SIZE_30 2>&1)

# values=$(extract_values "$output")
# success_rate=$(echo $values | cut -d',' -f1)
# avg_reward=$(echo $values | cut -d',' -f2)
# echo "30,$success_rate,$avg_reward" >> $RESULTS_FILE
# echo "✅ 30% - Success Rate: ${success_rate}%, Avg Reward: ${avg_reward}"

# # Test with 40% increase
# echo -e "\n📊 Testing with 40% size increase..."
# output=$(python tools/obj_geom_eval.py \
#   --run_dir $RUN_DIR \
#   --checkpoint $CHECKPOINT \
#   --morph $MORPH \
#   --task $TASK \
#   --controller $CONTROLLER \
#   --episodes $EPISODES \
#   --save_video ./test_varObject_vids/ \
#   --new-geom-size $SIZE_40 2>&1)

# values=$(extract_values "$output")
# success_rate=$(echo $values | cut -d',' -f1)
# avg_reward=$(echo $values | cut -d',' -f2)
# echo "40,$success_rate,$avg_reward" >> $RESULTS_FILE
# echo "✅ 40% - Success Rate: ${success_rate}%, Avg Reward: ${avg_reward}"

# # Test with 50% increase
# echo -e "\n📊 Testing with 50% size increase..."
# output=$(python tools/obj_geom_eval.py \
#   --run_dir $RUN_DIR \
#   --checkpoint $CHECKPOINT \
#   --morph $MORPH \
#   --task $TASK \
#   --controller $CONTROLLER \
#   --episodes $EPISODES \
#   --save_video ./test_varObject_vids/ \
#   --new-geom-size $SIZE_50 2>&1)

# values=$(extract_values "$output")
# success_rate=$(echo $values | cut -d',' -f1)
# avg_reward=$(echo $values | cut -d',' -f2)
# echo "50,$success_rate,$avg_reward" >> $RESULTS_FILE
# echo "✅ 50% - Success Rate: ${success_rate}%, Avg Reward: ${avg_reward}"

# # Test with 60% increase
# echo -e "\n📊 Testing with 60% size increase..."
# output=$(python tools/obj_geom_eval.py \
#   --run_dir $RUN_DIR \
#   --checkpoint $CHECKPOINT \
#   --morph $MORPH \
#   --task $TASK \
#   --controller $CONTROLLER \
#   --episodes $EPISODES \
#   --save_video ./test_varObject_vids/ \
#   --new-geom-size $SIZE_60 2>&1)

# values=$(extract_values "$output")
# success_rate=$(echo $values | cut -d',' -f1)
# avg_reward=$(echo $values | cut -d',' -f2)
# echo "60,$success_rate,$avg_reward" >> $RESULTS_FILE
# echo "✅ 60% - Success Rate: ${success_rate}%, Avg Reward: ${avg_reward}"

# # Test with 70% increase
# echo -e "\n📊 Testing with 70% size increase..."
# output=$(python tools/obj_geom_eval.py \
#   --run_dir $RUN_DIR \
#   --checkpoint $CHECKPOINT \
#   --morph $MORPH \
#   --task $TASK \
#   --controller $CONTROLLER \
#   --episodes $EPISODES \
#   --save_video ./test_varObject_vids/ \
#   --new-geom-size $SIZE_70 2>&1)

# values=$(extract_values "$output")
# success_rate=$(echo $values | cut -d',' -f1)
# avg_reward=$(echo $values | cut -d',' -f2)
# echo "70,$success_rate,$avg_reward" >> $RESULTS_FILE
# echo "✅ 70% - Success Rate: ${success_rate}%, Avg Reward: ${avg_reward}"

# # Test with 80% increase
# echo -e "\n📊 Testing with 80% size increase..."
# output=$(python tools/obj_geom_eval.py \
#   --run_dir $RUN_DIR \
#   --checkpoint $CHECKPOINT \
#   --morph $MORPH \
#   --task $TASK \
#   --controller $CONTROLLER \
#   --episodes $EPISODES \
#   --save_video ./test_varObject_vids/ \
#   --new-geom-size $SIZE_80 2>&1)

# values=$(extract_values "$output")
# success_rate=$(echo $values | cut -d',' -f1)
# avg_reward=$(echo $values | cut -d',' -f2)
# echo "80,$success_rate,$avg_reward" >> $RESULTS_FILE
# echo "✅ 80% - Success Rate: ${success_rate}%, Avg Reward: ${avg_reward}"

# # Test with 90% increase
# echo -e "\n📊 Testing with 90% size increase..."
# output=$(python tools/obj_geom_eval.py \
#   --run_dir $RUN_DIR \
#   --checkpoint $CHECKPOINT \
#   --morph $MORPH \
#   --task $TASK \
#   --controller $CONTROLLER \
#   --episodes $EPISODES \
#   --save_video ./test_varObject_vids/ \
#   --new-geom-size $SIZE_90 2>&1)

# values=$(extract_values "$output")
# success_rate=$(echo $values | cut -d',' -f1)
# avg_reward=$(echo $values | cut -d',' -f2)
# echo "90,$success_rate,$avg_reward" >> $RESULTS_FILE
# echo "✅ 90% - Success Rate: ${success_rate}%, Avg Reward: ${avg_reward}"

# # Test with 100% increase
# echo -e "\n📊 Testing with 100% size increase..."
# output=$(python tools/obj_geom_eval.py \
#   --run_dir $RUN_DIR \
#   --checkpoint $CHECKPOINT \
#   --morph $MORPH \
#   --task $TASK \
#   --controller $CONTROLLER \
#   --episodes $EPISODES \
#   --save_video ./test_varObject_vids/ \
#   --new-geom-size $SIZE_100 2>&1)

# values=$(extract_values "$output")
# success_rate=$(echo $values | cut -d',' -f1)
# avg_reward=$(echo $values | cut -d',' -f2)
# echo "100,$success_rate,$avg_reward" >> $RESULTS_FILE
# echo "✅ 100% - Success Rate: ${success_rate}%, Avg Reward: ${avg_reward}"

# echo -e "\n========================================="
# echo "Evaluation Complete! Results saved to $RESULTS_FILE"
# echo "========================================="

# # Display the collected data
# echo -e "\nCollected Data:"
# cat $RESULTS_FILE



#!/bin/bash

# --- 1. CONFIGURATION ---
# The only things you need to change are in this section.

# Path to the trained model's directory and the checkpoint file
RUN_DIR="./test_artifacts/Robosuite-v0-MR-ST-MR-MT_avg_nodes_1409-run:v19"
CHECKPOINT="checkpoint_600.pt"

# The specific SEEN morphology you want to test this generalization on
MORPH="Kinova3"

# The CUSTOM task variant you want to test (e.g., LiftScalableCube, LiftBall)
# This will be the environment that is physically simulated.
TASK="LiftScalableCube"

# The BASE task from which the learned skill/embedding should be drawn
BASE_TASK="Lift"

# Controller used for this robot-task pair
CONTROLLER="OSC_POSE"

# The seed used for training, to ensure evaluation is reproducible
SEED=1409

# Number of episodes to run for each scaling factor
EPISODES=5

# Directory to save videos (optional, can be left empty)
VIDEO_DIR="./analysis_results/geom_generalization_videos/"

# --- END OF CONFIGURATION ---


# --- 2. SCRIPT LOGIC (No changes needed below this line) ---

# Create a unique results file based on the config
RESULTS_FILE="results_${MORPH}_${TASK}_seed${SEED}.csv"
echo "Percentage_Increase,Scale_Factor,Success_Rate,Avg_Reward" > $RESULTS_FILE

echo "======================================================"
echo "Object Geometry Generalization Evaluation"
echo "Run: $RUN_DIR"
echo "Task: $TASK (using $BASE_TASK knowledge)"
echo "Morph: $MORPH"
echo "Seed: $SEED"
echo "Results will be saved to: $RESULTS_FILE"
echo "======================================================"

# Define the percentage increases you want to test
PERCENT_INCREASES=(0 20 40 50 90 100 120 140 150 200)

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

# Loop over each percentage increase
for percent in "${PERCENT_INCREASES[@]}"; do
    # Convert percentage to a scale factor (e.g., 20% -> 1.2)
    scale=$(echo "1 + $percent / 100" | bc -l)
    
    echo -e "\n📊 Testing with ${percent}% size increase (scale factor: ${scale})..."
    
    # Construct the command
    CMD="python tools/obj_geom_seeded.py \
      --run_dir $RUN_DIR \
      --checkpoint $CHECKPOINT \
      --morph $MORPH \
      --task $TASK \
      --base_task $BASE_TASK \
      --controller $CONTROLLER \
      --episodes $EPISODES \
      --save_video $VIDEO_DIR \
      --scale $scale"

    # # Add video saving argument only if the directory is specified
    # if [ ! -z "$VIDEO_DIR" ]; then
    #     CMD="$CMD --save_video $VIDEO_DIR"
    # fi

    # Execute the command and capture output
    output=$($CMD 2>&1)

    # Extract results from the output
    values=$(extract_values "$output")
    success_rate=$(echo $values | cut -d',' -f1)
    avg_reward=$(echo $values | cut -d',' -f2)
    
    # Save to CSV
    echo "$percent,$scale,$success_rate,$avg_reward" >> $RESULTS_FILE
    
    echo "✅ ${percent}% - Success Rate: ${success_rate}%, Avg Reward: ${avg_reward}"
done

echo -e "\n======================================================"
echo "Evaluation Complete! Results saved to $RESULTS_FILE"
echo "======================================================"

# Display the collected data
echo -e "\nCollected Data:"
cat $RESULTS_FILE

# The plotting logic can remain the same or be a separate Python script
# For simplicity, here's how you could call a plotting script
# python plot_results.py --file $RESULTS_FILE

# Create plots using embedded Python script
echo -e "\n📈 Generating plots..."
python3 << 'EOF'
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read the results
try:
    df = pd.read_csv('size_generalization_results.txt')
    
    print("\nData loaded successfully:")
    print(df)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Success Rate vs Size Increase
    ax1.plot(df['Size_Percent'], df['Success_Rate'], 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Object Size Increase (%)', fontsize=12)
    ax1.set_ylabel('Success Rate (%)', fontsize=12)
    ax1.set_title('Success Rate vs Object Size Increase', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([max(0, df['Success_Rate'].min()-5), min(105, df['Success_Rate'].max()+5)])
    ax1.set_xlim([-2, 27])
    
    # Add value labels on points
    for x, y in zip(df['Size_Percent'], df['Success_Rate']):
        ax1.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=9)
    
    # Plot 2: Average Reward vs Size Increase
    ax2.plot(df['Size_Percent'], df['Avg_Reward'], 's-', linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel('Object Size Increase (%)', fontsize=12)
    ax2.set_ylabel('Average Reward', fontsize=12)
    ax2.set_title('Average Reward vs Object Size Increase', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-2, 27])
    
    # Add value labels on points
    for x, y in zip(df['Size_Percent'], df['Avg_Reward']):
        ax2.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=9)
    
    # Get task and morph from the first line of the shell script output if possible
    task = "Door"  # Default, will be replaced if found
    morph = "Sawyer"  # Default, will be replaced if found
    
    # Add a main title
    fig.suptitle(f'Generalization to Different Object Sizes\nTask: {task}, Morph: {morph}', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('object_size_generalization_plots.png', dpi=150, bbox_inches='tight')
    print("✅ Plots saved to 'object_size_generalization_plots.png'")
    
    # Also create individual plots for better clarity
    # Individual plot for Success Rate
    fig1, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df['Size_Percent'], df['Success_Rate'], 'o-', linewidth=2.5, markersize=10, color='#2E86AB')
    ax.fill_between(df['Size_Percent'], df['Success_Rate'], alpha=0.3, color='#2E86AB')
    ax.set_xlabel('Object Size Increase (%)', fontsize=14)
    ax.set_ylabel('Success Rate (%)', fontsize=14)
    ax.set_title('Success Rate vs Object Size Increase', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([max(0, df['Success_Rate'].min()-5), min(105, df['Success_Rate'].max()+5)])
    ax.set_xlim([-2, 27])
    
    for x, y in zip(df['Size_Percent'], df['Success_Rate']):
        ax.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", xytext=(0,8), ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('success_rate_vs_size.png', dpi=150, bbox_inches='tight')
    print("✅ Success rate plot saved to 'success_rate_vs_size.png'")
    
    # Individual plot for Average Reward
    fig2, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df['Size_Percent'], df['Avg_Reward'], 's-', linewidth=2.5, markersize=10, color='#A23B72')
    ax.fill_between(df['Size_Percent'], df['Avg_Reward'], alpha=0.3, color='#A23B72')
    ax.set_xlabel('Object Size Increase (%)', fontsize=14)
    ax.set_ylabel('Average Reward', fontsize=14)
    ax.set_title('Average Reward vs Object Size Increase', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([-2, 27])
    
    # Adjust y-axis to show the data better
    y_min, y_max = df['Avg_Reward'].min(), df['Avg_Reward'].max()
    y_range = y_max - y_min
    ax.set_ylim([y_min - 0.1*y_range, y_max + 0.1*y_range])
    
    for x, y in zip(df['Size_Percent'], df['Avg_Reward']):
        ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,8), ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('avg_reward_vs_size.png', dpi=150, bbox_inches='tight')
    print("✅ Average reward plot saved to 'avg_reward_vs_size.png'")
    
    # Print summary statistics
    print("\n📊 Summary Statistics:")
    print("="*40)
    print(f"Original (0%) Success Rate: {df.loc[df['Size_Percent']==0, 'Success_Rate'].values[0]:.1f}%")
    print(f"25% Increase Success Rate: {df.loc[df['Size_Percent']==25, 'Success_Rate'].values[0]:.1f}%")
    print(f"Success Rate Drop: {df.loc[df['Size_Percent']==0, 'Success_Rate'].values[0] - df.loc[df['Size_Percent']==25, 'Success_Rate'].values[0]:.1f}%")
    print("-"*40)
    print(f"Original (0%) Avg Reward: {df.loc[df['Size_Percent']==0, 'Avg_Reward'].values[0]:.2f}")
    print(f"25% Increase Avg Reward: {df.loc[df['Size_Percent']==25, 'Avg_Reward'].values[0]:.2f}")
    print(f"Reward Drop: {df.loc[df['Size_Percent']==0, 'Avg_Reward'].values[0] - df.loc[df['Size_Percent']==25, 'Avg_Reward'].values[0]:.2f}")
    print("="*40)
    
except Exception as e:
    print(f"Error creating plots: {e}")
    import traceback
    traceback.print_exc()
    print("Make sure matplotlib and pandas are installed: pip install matplotlib pandas")
EOF

echo -e "\n✨ All done! Check the generated plots and results file."