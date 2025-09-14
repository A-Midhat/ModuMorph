# #!/usr/bin/env bash
# set -euo pipefail

# # =============================================================================
# # FULL EVALUATION PIPELINE FOR MULTI-ROBOT MULTI-TASK GENERALIZATION
# # =============================================================================
# # This script runs all evaluations and aggregates results for paper reporting
# # Output: final_results.csv with mean±std for each (morph, variant, task)

# # --- CONFIGURATION ---
# MAX_JOBS=6                    # Concurrent jobs (adjust based on your hardware)
# EPISODES=1                   # Episodes per evaluation
# CHECKPOINT="Robosuite-v0.pt" # Checkpoint filename
# CONTROLLER="OSC_POSE"         # Controller type
# RESULT_ROOT="./test_results_by_type"
# FINAL_RESULTS_DIR="./final_paper_results"

# # Create directories
# mkdir -p "$RESULT_ROOT" "$FINAL_RESULTS_DIR"

# # --- EXPERIMENT CONFIGURATION ---
# # Update these paths to match your actual artifact locations
# declare -A ARTIFACT_PATHS=(
#     # AllNodes variants (3 seeds each)
#     ["allnodes_seed1"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_ALLNODES4_1409-run:v9"
#     ["allnodes_seed2"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_ALLNODES4_3296-run:v9"  
#     ["allnodes_seed3"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_ALLNODES4_2008-run:v9"
    
#     # Avg_nodes variants (3 seeds each)
#     ["avg_nodes_seed1"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_avg_nodes_1409-run:v19"
#     ["avg_nodes_seed2"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_avg_nodes_3296-run:v18"
#     ["avg_nodes_seed3"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_avg_nodes_2008-run:v18"
    
#     # Object_only variants (3 seeds each)  
#     ["object_only_seed1"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_object_only_1409-run:v20"
#     ["object_only_seed2"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_object_only_3296-run:v18"
#     ["object_only_seed3"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_object_only4_2008-run:v9"
# )
# #  "PickPlaceCan" "Wipe"
# # "Panda" 
# MORPHS=("UR5e")
# TASKS=("Door" "Lift")

# # Function to extract variant type from key
# get_variant_type() {
#     local key="$1"
#     if [[ $key == allnodes_* ]]; then
#         echo "allnodes"
#     elif [[ $key == avg_nodes_* ]]; then
#         echo "avg_nodes"
#     elif [[ $key == object_only_* ]]; then
#         echo "object_only"
#     else
#         echo "unknown"
#     fi
# }

# # Function to extract seed number from key
# get_seed_number() {
#     local key="$1"
#     if [[ $key =~ seed([0-9]+)$ ]]; then
#         echo "${BASH_REMATCH[1]}"
#     else
#         echo "unknown"
#     fi
# }

# # --- SEMAPHORE SETUP FOR CONCURRENCY CONTROL ---
# fifo="/tmp/sem.$$"
# mkfifo "$fifo"
# exec 9<> "$fifo"
# rm "$fifo"

# # Fill semaphore with MAX_JOBS tokens
# for ((i=0; i<MAX_JOBS; i++)); do
#     echo >&9
# done

# # --- PHASE 1: RUN ALL EVALUATIONS ---
# echo "=========================================="
# echo "PHASE 1: Running Individual Evaluations"
# echo "=========================================="
# echo "Total configurations: $((${#ARTIFACT_PATHS[@]} * ${#MORPHS[@]} * ${#TASKS[@]}))"
# echo "Episodes per evaluation: $EPISODES"
# echo "Max concurrent jobs: $MAX_JOBS"
# echo ""

# job_counter=1
# total_jobs=$((${#ARTIFACT_PATHS[@]} * ${#MORPHS[@]} * ${#TASKS[@]}))

# for key in "${!ARTIFACT_PATHS[@]}"; do
#     artifact_path="${ARTIFACT_PATHS[$key]}"
#     variant_type=$(get_variant_type "$key")
#     seed_num=$(get_seed_number "$key")
    
#     for morph in "${MORPHS[@]}"; do
#         for task in "${TASKS[@]}"; do
#             echo "Queueing job $job_counter/$total_jobs: $key | $morph | $task"
#             ((job_counter++))
            
#             # Block until token available
#             read -u 9 || true
            
#             {
#                 set -e
#                 artifact_basename=$(basename "$artifact_path")
#                 out_dir="${RESULT_ROOT}/${variant_type}"
#                 mkdir -p "$out_dir"
                
#                 echo "[START] $(date +'%H:%M:%S') $key -> $morph/$task"
                
#                 python tools/zs_morph_eval.py \
#                     --run_dir "./$artifact_path/" \
#                     --checkpoint "$CHECKPOINT" \
#                     --morph "$morph" \
#                     --task "$task" \
#                     --controller "$CONTROLLER" \
#                     --episodes "$EPISODES" \
#                     --save_video "$out_dir/videos_${artifact_basename}_${morph}_${task}/" \
#                     --save_metrics \
#                     --out_dir "$out_dir" \
#                     || echo "[FAILED] $key -> $morph/$task"
                
#                 echo "[DONE] $(date +'%H:%M:%S') $key -> $morph/$task"
                
#                 # Release token
#                 echo >&9
#             } &
#         done
#     done
# done

# echo "Waiting for all evaluation jobs to complete..."
# wait
# exec 9>&-

# echo ""
# echo "PHASE 1 COMPLETE: All individual evaluations finished"
# echo ""

# # --- PHASE 2: AGGREGATE RESULTS ---
# echo "=========================================="
# echo "PHASE 2: Aggregating Results Across Seeds"
# echo "=========================================="

# # Create Python aggregation script
# cat > "${FINAL_RESULTS_DIR}/aggregate_results.py" << 'EOF'
# #!/usr/bin/env python3
# """
# Aggregate evaluation results across seeds for paper reporting.
# """
# import os
# import json
# import pandas as pd
# import numpy as np
# from collections import defaultdict
# import argparse

# def load_all_results(results_root):
#     """Load all JSON results from the directory structure."""
#     all_results = []
    
#     # Walk through results_by_type structure
#     for variant_dir in os.listdir(results_root):
#         variant_path = os.path.join(results_root, variant_dir)
#         if not os.path.isdir(variant_path):
#             continue
            
#         variant_type = variant_dir  # allnodes, avg_nodes, object_only
        
#         # Walk through artifact directories
#         for artifact_dir in os.listdir(variant_path):
#             artifact_path = os.path.join(variant_path, artifact_dir)
#             if not os.path.isdir(artifact_path):
#                 continue
                
#             # Look for JSON files
#             for filename in os.listdir(artifact_path):
#                 if filename.endswith('_metrics.json'):
#                     json_path = os.path.join(artifact_path, filename)
#                     try:
#                         with open(json_path, 'r') as f:
#                             data = json.load(f)
                        
#                         # Extract seed from artifact name or filename
#                         seed = extract_seed_from_name(data.get('artifact', artifact_dir))
                        
#                         result = {
#                             'variant': variant_type,
#                             'seed': seed,
#                             'morph': data.get('morph'),
#                             'task': data.get('task'), 
#                             'avg_reward': data.get('avg_reward'),
#                             'std_reward': data.get('std_reward'),
#                             'success_rate_pct': data.get('success_rate_pct'),
#                             'episodes': data.get('episodes'),
#                             'artifact': data.get('artifact', artifact_dir)
#                         }
#                         all_results.append(result)
                        
#                     except Exception as e:
#                         print(f"Error loading {json_path}: {e}")
    
#     return all_results

# def extract_seed_from_name(name):
#     """Extract seed number from artifact name."""
#     import re
#     # Look for patterns like _1409-, _3296-, _2008-
#     match = re.search(r'_(\d{4})-', name)
#     if match:
#         return match.group(1)
#     return 'unknown'

# def aggregate_by_groups(results_df):
#     """Aggregate results by (morph, variant, task) groups."""
    
#     # Group by morph, variant, task
#     grouped = results_df.groupby(['morph', 'variant', 'task'])
    
#     aggregated_results = []
    
#     for (morph, variant, task), group in grouped:
#         if len(group) == 0:
#             continue
            
#         # Calculate statistics across seeds
#         mean_reward = group['avg_reward'].mean()
#         std_reward = group['avg_reward'].std(ddof=1) if len(group) > 1 else 0
        
#         mean_success = group['success_rate_pct'].mean()
#         std_success = group['success_rate_pct'].std(ddof=1) if len(group) > 1 else 0
        
#         n_seeds = len(group)
#         seeds_used = sorted(group['seed'].unique())
        
#         # Calculate 95% confidence interval for mean reward
#         if n_seeds > 1:
#             sem_reward = std_reward / np.sqrt(n_seeds)
#             ci_reward = 1.96 * sem_reward  # 95% CI
#         else:
#             ci_reward = 0
            
#         result = {
#             'morph': morph,
#             'variant': variant, 
#             'task': task,
#             'n_seeds': n_seeds,
#             'seeds_used': ','.join(map(str, seeds_used)),
#             'mean_reward': mean_reward,
#             'std_reward': std_reward,
#             'ci95_reward': ci_reward,
#             'mean_success_rate': mean_success,
#             'std_success_rate': std_success,
#             'formatted_reward': f"{mean_reward:.1f} ± {std_reward:.1f}",
#             'formatted_success': f"{mean_success:.1f} ± {std_success:.1f}%"
#         }
        
#         aggregated_results.append(result)
    
#     return pd.DataFrame(aggregated_results)

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--results_root', required=True)
#     parser.add_argument('--output_dir', required=True) 
#     args = parser.parse_args()
    
#     print(f"Loading results from: {args.results_root}")
    
#     # Load all individual results
#     all_results = load_all_results(args.results_root)
#     if not all_results:
#         print("No results found!")
#         return
        
#     results_df = pd.DataFrame(all_results)
#     print(f"Loaded {len(results_df)} individual evaluation results")
    
#     # Save individual results
#     individual_path = os.path.join(args.output_dir, 'individual_results.csv')
#     results_df.to_csv(individual_path, index=False)
#     print(f"Saved individual results to: {individual_path}")
    
#     # Print summary of what we found
#     print("\nData Summary:")
#     print(f"Variants: {sorted(results_df['variant'].unique())}")
#     print(f"Morphs: {sorted(results_df['morph'].unique())}")
#     print(f"Tasks: {sorted(results_df['task'].unique())}")
#     print(f"Seeds found: {sorted(results_df['seed'].unique())}")
    
#     # Aggregate across seeds
#     print("\nAggregating across seeds...")
#     aggregated_df = aggregate_by_groups(results_df)
    
#     # Save aggregated results
#     agg_path = os.path.join(args.output_dir, 'final_results.csv')
#     aggregated_df.to_csv(agg_path, index=False)
#     print(f"Saved final aggregated results to: {agg_path}")
    
#     # Create paper-ready table
#     print("\n" + "="*80)
#     print("PAPER-READY RESULTS")
#     print("="*80)
    
#     # Pivot table for easy reading
#     for morph in sorted(aggregated_df['morph'].unique()):
#         morph_data = aggregated_df[aggregated_df['morph'] == morph]
        
#         print(f"\n--- {morph.upper()} RESULTS ---")
#         print(f"{'Variant':<15} {'Task':<15} {'Reward (Mean±Std)':<20} {'Success Rate (%)':<20} {'N Seeds':<8}")
#         print("-" * 80)
        
#         for _, row in morph_data.sort_values(['variant', 'task']).iterrows():
#             print(f"{row['variant']:<15} {row['task']:<15} {row['formatted_reward']:<20} {row['formatted_success']:<20} {row['n_seeds']:<8}")
    
#     print("\n" + "="*80)
#     print(f"Results saved to: {agg_path}")
#     print("Individual results saved to:", individual_path)

# if __name__ == '__main__':
#     main()
# EOF

# # Run the aggregation script
# echo "Running result aggregation..."
# python3 "${FINAL_RESULTS_DIR}/aggregate_results.py" \
#     --results_root "$RESULT_ROOT" \
#     --output_dir "$FINAL_RESULTS_DIR"

# echo ""
# echo "=========================================="
# echo "PIPELINE COMPLETE!"
# echo "=========================================="
# echo "Final results available in: $FINAL_RESULTS_DIR"
# echo "  - final_results.csv: Aggregated results for paper"
# echo "  - individual_results.csv: All individual runs"
# echo ""
# echo "Check the terminal output above for paper-ready formatted results!"
# echo "=========================================="



############################################################################
#!/usr/bin/env bash
# set -euo pipefail

# # =============================================================================
# # FULL EVALUATION PIPELINE WITH IMPROVED ERROR HANDLING
# # =============================================================================

# # --- CONFIGURATION ---
# MAX_JOBS=3                    # Reduce concurrency to avoid resource issues
# EPISODES=1                   
# CHECKPOINT="Robosuite-v0.pt" 
# CONTROLLER="OSC_POSE"         
# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# RESULT_ROOT="./test_results_by_type_${TIMESTAMP}"
# FINAL_RESULTS_DIR="./final_paper_results_${TIMESTAMP}"
# LOG_DIR="./evaluation_logs_${TIMESTAMP}"

# # Clean any existing non-timestamped directories to avoid confusion
# if [[ -d "./test_results_by_type" ]]; then
#     echo "Cleaning previous results directory..."
#     rm -rf "./test_results_by_type"
# fi

# # Create directories
# mkdir -p "$RESULT_ROOT" "$FINAL_RESULTS_DIR" "$LOG_DIR"

# # --- EXPERIMENT CONFIGURATION ---
# declare -A ARTIFACT_PATHS=(
#     ["allnodes_seed1"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_ALLNODES4_1409-run:v9"
#     ["allnodes_seed2"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_ALLNODES4_3296-run:v9"  
#     ["allnodes_seed3"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_ALLNODES4_2008-run:v9"
#     ["avg_nodes_seed1"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_avg_nodes_1409-run:v19"
#     ["avg_nodes_seed2"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_avg_nodes_3296-run:v18"
#     ["avg_nodes_seed3"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_avg_nodes_2008-run:v18"
#     ["object_only_seed1"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_object_only_1409-run:v20"
#     ["object_only_seed2"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_object_only_3296-run:v18"
#     ["object_only_seed3"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_object_only4_2008-run:v9"
# )

# MORPHS=("UR5e")
# TASKS=("Door" "Lift")

# # Function to extract variant type from key
# get_variant_type() {
#     local key="$1"
#     if [[ $key == allnodes_* ]]; then
#         echo "allnodes"
#     elif [[ $key == avg_nodes_* ]]; then
#         echo "avg_nodes"
#     elif [[ $key == object_only_* ]]; then
#         echo "object_only"
#     else
#         echo "unknown"
#     fi
# }

# # Function to extract seed number from key
# get_seed_number() {
#     local key="$1"
#     if [[ $key =~ seed([0-9]+)$ ]]; then
#         echo "${BASH_REMATCH[1]}"
#     else
#         echo "unknown"
#     fi
# }

# echo "=========================================="
# echo "PHASE 0: Pre-flight Checks"
# echo "=========================================="

# # Check if all artifacts and checkpoints exist
# failed_checks=0
# for key in "${!ARTIFACT_PATHS[@]}"; do
#     artifact_path="${ARTIFACT_PATHS[$key]}"
#     checkpoint_path="./${artifact_path}/${CHECKPOINT}"
    
#     if [[ ! -d "./$artifact_path" ]]; then
#         echo "❌ MISSING ARTIFACT DIR: $key -> ./$artifact_path"
#         ((failed_checks++))
#         continue
#     fi
    
#     if [[ ! -f "$checkpoint_path" ]]; then
#         echo "❌ MISSING CHECKPOINT: $key -> $checkpoint_path"
#         echo "   Available files in ./$artifact_path:"
#         ls -la "./$artifact_path" | grep -E '\.(pt|pth)$' || echo "   No .pt/.pth files found"
#         ((failed_checks++))
#         continue
#     fi
    
#     echo "✅ VALID: $key"
# done

# if [[ $failed_checks -gt 0 ]]; then
#     echo "❌ $failed_checks artifact(s) failed pre-flight checks!"
#     echo "Please fix the missing artifacts/checkpoints before continuing."
#     echo "Press Enter to continue anyway or Ctrl+C to abort."
#     read -r
# fi

# echo "Pre-flight check complete."
# echo ""

# # --- SEMAPHORE SETUP FOR CONCURRENCY CONTROL ---
# fifo="/tmp/sem.$$"
# mkfifo "$fifo"
# exec 9<> "$fifo"
# rm "$fifo"

# # Fill semaphore with MAX_JOBS tokens
# for ((i=0; i<MAX_JOBS; i++)); do
#     echo >&9
# done

# # --- PHASE 1: RUN ALL EVALUATIONS ---
# echo "=========================================="
# echo "PHASE 1: Running Individual Evaluations"
# echo "=========================================="
# echo "Total configurations: $((${#ARTIFACT_PATHS[@]} * ${#MORPHS[@]} * ${#TASKS[@]}))"
# echo "Episodes per evaluation: $EPISODES"
# echo "Max concurrent jobs: $MAX_JOBS"
# echo ""

# job_counter=1
# total_jobs=$((${#ARTIFACT_PATHS[@]} * ${#MORPHS[@]} * ${#TASKS[@]}))
# declare -a job_pids=()
# declare -a job_descriptions=()

# for key in "${!ARTIFACT_PATHS[@]}"; do
#     artifact_path="${ARTIFACT_PATHS[$key]}"
#     variant_type=$(get_variant_type "$key")
#     seed_num=$(get_seed_number "$key")
    
#     for morph in "${MORPHS[@]}"; do
#         for task in "${TASKS[@]}"; do
#             job_desc="$key|$morph|$task"
#             echo "Queueing job $job_counter/$total_jobs: $job_desc"
            
#             # Block until token available
#             read -u 9 || true
            
#             {
#                 set -e
#                 artifact_basename=$(basename "$artifact_path")
#                 out_dir="${RESULT_ROOT}/${variant_type}"
#                 mkdir -p "$out_dir"
                
#                 # Create detailed log file for this job
#                 log_file="${LOG_DIR}/${key}_${morph}_${task}.log"
                
#                 echo "[START] $(date +'%H:%M:%S') $job_desc" | tee -a "$log_file"
                
#                 # Run the evaluation with comprehensive logging
#                 if python tools/zs_morph_eval.py \
#                     --run_dir "./$artifact_path/" \
#                     --checkpoint "$CHECKPOINT" \
#                     --morph "$morph" \
#                     --task "$task" \
#                     --controller "$CONTROLLER" \
#                     --episodes "$EPISODES" \
#                     --save_video "$out_dir/videos_${artifact_basename}_${morph}_${task}/" \
#                     --save_metrics \
#                     --out_dir "$out_dir" \
#                     >> "$log_file" 2>&1; then
                    
#                     echo "[SUCCESS] $(date +'%H:%M:%S') $job_desc" | tee -a "$log_file"
#                 else
#                     echo "[FAILED] $(date +'%H:%M:%S') $job_desc - Check $log_file for details" | tee -a "$log_file"
#                 fi
                
#                 # Release token
#                 echo >&9
#             } &
            
#             job_pids+=($!)
#             job_descriptions+=("$job_desc")
#             ((job_counter++))
#         done
#     done
# done

# echo "Waiting for all evaluation jobs to complete..."

# # Wait for all jobs and check their status
# failed_jobs=()
# for i in "${!job_pids[@]}"; do
#     pid=${job_pids[i]}
#     desc=${job_descriptions[i]}
    
#     if wait "$pid"; then
#         echo "✅ Job completed successfully: $desc"
#     else
#         echo "❌ Job failed: $desc"
#         failed_jobs+=("$desc")
#     fi
# done

# exec 9>&-

# echo ""
# if [[ ${#failed_jobs[@]} -gt 0 ]]; then
#     echo "❌ ${#failed_jobs[@]} job(s) failed:"
#     printf '   - %s\n' "${failed_jobs[@]}"
#     echo ""
#     echo "Check log files in $LOG_DIR for detailed error information."
#     echo ""
# fi

# echo "PHASE 1 COMPLETE: All individual evaluations finished"
# echo "Check $LOG_DIR for individual job logs"
# echo ""

# # # --- PHASE 2: AGGREGATE RESULTS ---
# # echo "=========================================="
# # echo "PHASE 2: Aggregating Results Across Seeds"
# # echo "=========================================="

# # # Use the same Python aggregation script as before
# # python3 "${FINAL_RESULTS_DIR}/aggregate_results.py" \
# #     --results_root "$RESULT_ROOT" \
# #     --output_dir "$FINAL_RESULTS_DIR"
# # --- PHASE 2: AGGREGATE RESULTS ---
# echo "=========================================="
# echo "PHASE 2: Aggregating Results Across Seeds"
# echo "=========================================="

# # Use the permanent aggregation script
# python3 "./aggregate_results.py" \
#     --results_root "$RESULT_ROOT" \
#     --output_dir "$FINAL_RESULTS_DIR"

# echo ""
# echo "=========================================="
# echo "PIPELINE COMPLETE!"
# echo "=========================================="
# echo "Final results available in: $FINAL_RESULTS_DIR"
# echo "Individual job logs available in: $LOG_DIR"
# if [[ ${#failed_jobs[@]} -gt 0 ]]; then
#     echo "❌ WARNING: ${#failed_jobs[@]} jobs failed - results may be incomplete"
# fi
# echo "=========================================="



############################################################################
#!/usr/bin/env bash
set -euo pipefail

# SIMPLE EVALUATION SCRIPT - EVERYTHING IN ONE PLACE
MAX_JOBS=3
EPISODES=64
CHECKPOINT="checkpoint_600.pt"
CONTROLLER="OSC_POSE"
RESULT_ROOT="./results"
rm -rf "$RESULT_ROOT"  # Clean start
mkdir -p "$RESULT_ROOT"

declare -A ARTIFACT_PATHS=(
    ["allnodes_seed1"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_ALLNODES4_1409-run:v9"
    ["allnodes_seed2"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_ALLNODES4_3296-run:v9"  
    ["allnodes_seed3"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_ALLNODES4_2008-run:v9"
    ["avg_nodes_seed1"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_avg_nodes_1409-run:v19"
    ["avg_nodes_seed2"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_avg_nodes_3296-run:v18"
    ["avg_nodes_seed3"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_avg_nodes_2008-run:v18"
    ["object_only_seed1"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_object_only_1409-run:v20"
    ["object_only_seed2"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_object_only_3296-run:v18"
    ["object_only_seed3"]="test_artifacts/Robosuite-v0-MR-ST-MR-MT_object_only4_2008-run:v9"
)

MORPHS=("UR5e")
TASKS=("Door" "Lift" "PickPlaceCan" "Wipe")

# RUN EVALUATIONS
echo "Running evaluations..."
counter=1
total=$((${#ARTIFACT_PATHS[@]} * ${#MORPHS[@]} * ${#TASKS[@]}))

for key in "${!ARTIFACT_PATHS[@]}"; do
    artifact_path="${ARTIFACT_PATHS[$key]}"
    for morph in "${MORPHS[@]}"; do
        for task in "${TASKS[@]}"; do
            echo "[$counter/$total] $key -> $morph/$task"
            
            python tools/zs_morph_eval.py \
                --run_dir "./$artifact_path/" \
                --checkpoint "$CHECKPOINT" \
                --morph "$morph" \
                --task "$task" \
                --controller "$CONTROLLER" \
                --episodes "$EPISODES" \
                --save_video "$RESULT_ROOT/videos_${key}_${morph}_${task}/" \
                --save_metrics \
                --out_dir "$RESULT_ROOT" || echo "FAILED: $key -> $morph/$task"
            
            ((counter++))
        done
    done
done
# AGGREGATE RESULTS
echo ""
echo "Aggregating results..."

# Create simple aggregation script inline
python3 << 'EOF'
import os
import json
import pandas as pd
import numpy as np

def extract_seed(name):
    import re
    match = re.search(r'_(\d{4})-', name)
    return match.group(1) if match else 'unknown'

def detect_type(name):
    name = name.lower()
    if 'allnodes' in name: return 'allnodes'
    elif 'avg_nodes' in name or 'avgnodes' in name: return 'avg_nodes'
    elif 'object_only' in name or 'objectonly' in name: return 'object_only'
    else: return 'unknown'

# Load all JSON files
results = []
result_dir = "./results"

for subdir in os.listdir(result_dir):
    subdir_path = os.path.join(result_dir, subdir)
    if not os.path.isdir(subdir_path):
        continue
        
    for filename in os.listdir(subdir_path):
        if filename.endswith('_metrics.json'):
            filepath = os.path.join(subdir_path, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                results.append({
                    'variant': detect_type(data.get('artifact', filename)),
                    'seed': extract_seed(data.get('artifact', filename)),
                    'morph': data.get('morph'),
                    'task': data.get('task'),
                    'avg_reward': data.get('avg_reward'),
                    'success_rate_pct': data.get('success_rate_pct')
                })
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

if not results:
    print("No results found!")
    exit(1)

df = pd.DataFrame(results)
print(f"Loaded {len(df)} results")

# Aggregate by variant/task
grouped = df.groupby(['variant', 'task']).agg({
    'avg_reward': ['mean', 'std'],
    'success_rate_pct': ['mean', 'std'],
    'seed': 'count'
}).round(1)

# --- START OF FIX ---
# Flatten the multi-level column index to avoid shell syntax clashes
grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
grouped = grouped.reset_index()

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"{'Variant':<15} {'Task':<8} {'Reward':<15} {'Success %':<15} {'N Seeds':<8}")
print("-" * 70)

for index, row in grouped.iterrows():
    variant = row['variant']
    task = row['task']
    reward_mean = row['avg_reward_mean']
    reward_std = row['avg_reward_std']
    success_mean = row['success_rate_pct_mean']
    success_std = row['success_rate_pct_std']
    n_seeds = int(row['seed_count'])
    
    reward_str = f"{reward_mean:.1f} ± {reward_std:.1f}" if not pd.isna(reward_std) else f"{reward_mean:.1f}"
    success_str = f"{success_mean:.1f} ± {success_std:.1f}" if not pd.isna(success_std) else f"{success_mean:.1f}"
    
    print(f"{variant:<15} {task:<8} {reward_str:<15} {success_str:<15} {n_seeds:<8}")

# --- END OF FIX ---

print("="*70)

# Save detailed results
df.to_csv('./final_results.csv', index=False)
print("Detailed results saved to: ./final_results.csv")
EOF

echo "DONE! Check ./final_results.csv for detailed results."