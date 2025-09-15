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



# ############################################################################
# #!/usr/bin/env bash
# set -euo pipefail

# # SIMPLE EVALUATION SCRIPT - EVERYTHING IN ONE PLACE
# MAX_JOBS=3
# EPISODES=64
# CHECKPOINT="checkpoint_600.pt"
# CONTROLLER="OSC_POSE"
# RESULT_ROOT="./results"
# rm -rf "$RESULT_ROOT"  # Clean start
# mkdir -p "$RESULT_ROOT"

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

# MORPHS=("UR5e" "Panda")
# TASKS=("Door" "Lift" "PickPlaceCan" "Wipe")

# # RUN EVALUATIONS
# echo "Running evaluations..."
# counter=1
# total=$((${#ARTIFACT_PATHS[@]} * ${#MORPHS[@]} * ${#TASKS[@]}))

# for key in "${!ARTIFACT_PATHS[@]}"; do
#     artifact_path="${ARTIFACT_PATHS[$key]}"
#     for morph in "${MORPHS[@]}"; do
#         for task in "${TASKS[@]}"; do
#             echo "[$counter/$total] $key -> $morph/$task"
            
#             python tools/zs_morph_eval.py \
#                 --run_dir "./$artifact_path/" \
#                 --checkpoint "$CHECKPOINT" \
#                 --morph "$morph" \
#                 --task "$task" \
#                 --controller "$CONTROLLER" \
#                 --episodes "$EPISODES" \
#                 --save_video "$RESULT_ROOT/videos_${key}_${morph}_${task}/" \
#                 --save_metrics \
#                 --out_dir "$RESULT_ROOT" || echo "FAILED: $key -> $morph/$task"
            
#             ((counter++))
#         done
#     done
# done
# # AGGREGATE RESULTS
# echo ""
# echo "Aggregating results..."

# # Create simple aggregation script inline
# python3 << 'EOF'
# import os
# import json
# import pandas as pd
# import numpy as np

# def extract_seed(name):
#     import re
#     match = re.search(r'_(\d{4})-', name)
#     return match.group(1) if match else 'unknown'

# def detect_type(name):
#     name = name.lower()
#     if 'allnodes' in name: return 'allnodes'
#     elif 'avg_nodes' in name or 'avgnodes' in name: return 'avg_nodes'
#     elif 'object_only' in name or 'objectonly' in name: return 'object_only'
#     else: return 'unknown'

# # Load all JSON files
# results = []
# result_dir = "./results"

# for subdir in os.listdir(result_dir):
#     subdir_path = os.path.join(result_dir, subdir)
#     if not os.path.isdir(subdir_path):
#         continue
        
#     for filename in os.listdir(subdir_path):
#         if filename.endswith('_metrics.json'):
#             filepath = os.path.join(subdir_path, filename)
#             try:
#                 with open(filepath, 'r') as f:
#                     data = json.load(f)
                
#                 results.append({
#                     'variant': detect_type(data.get('artifact', filename)),
#                     'seed': extract_seed(data.get('artifact', filename)),
#                     'morph': data.get('morph'),
#                     'task': data.get('task'),
#                     'avg_reward': data.get('avg_reward'),
#                     'success_rate_pct': data.get('success_rate_pct')
#                 })
#             except Exception as e:
#                 print(f"Error loading {filepath}: {e}")

# if not results:
#     print("No results found!")
#     exit(1)

# df = pd.DataFrame(results)
# print(f"Loaded {len(df)} results")

# # Aggregate by variant/task
# grouped = df.groupby(['variant', 'task']).agg({
#     'avg_reward': ['mean', 'std'],
#     'success_rate_pct': ['mean', 'std'],
#     'seed': 'count'
# }).round(1)

# # --- START OF FIX ---
# # Flatten the multi-level column index to avoid shell syntax clashes
# grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
# grouped = grouped.reset_index()

# print("\n" + "="*70)
# print("FINAL RESULTS")
# print("="*70)
# print(f"{'Variant':<15} {'Task':<8} {'Reward':<15} {'Success %':<15} {'N Seeds':<8}")
# print("-" * 70)

# for index, row in grouped.iterrows():
#     variant = row['variant']
#     task = row['task']
#     reward_mean = row['avg_reward_mean']
#     reward_std = row['avg_reward_std']
#     success_mean = row['success_rate_pct_mean']
#     success_std = row['success_rate_pct_std']
#     n_seeds = int(row['seed_count'])
    
#     reward_str = f"{reward_mean:.1f} ± {reward_std:.1f}" if not pd.isna(reward_std) else f"{reward_mean:.1f}"
#     success_str = f"{success_mean:.1f} ± {success_std:.1f}" if not pd.isna(success_std) else f"{success_mean:.1f}"
    
#     print(f"{variant:<15} {task:<8} {reward_str:<15} {success_str:<15} {n_seeds:<8}")

# # --- END OF FIX ---

# print("="*70)

# # Save detailed results
# df.to_csv('./final_results.csv', index=False)
# print("Detailed results saved to: ./final_results.csv")
# EOF

# echo "DONE! Check ./final_results.csv for detailed results."




#!/usr/bin/env bash
set -euo pipefail

####################################################################
####################### Enhanced Parallel Script ##################
####################################################################

# Check if GNU parallel is installed
if ! command -v parallel &> /dev/null; then
    echo "⚠️  GNU Parallel not found. Installing..."
    sudo apt-get update && sudo apt-get install -y parallel
    echo "✅ GNU Parallel installed successfully"
fi

############################################
# Enhanced Parallel unseen-morph evaluation launcher
# - Calls tools/zs_morph_eval.py in parallel
# - Multi-terminal safety with unique run IDs
# - Statistical aggregation across seeds
# - Professional visualization and analysis
# Usage: source venv/bin/activate && ./tools/enhanced_parallel_zs_eval.sh
############################################

# -------------------------
# User-configurable options
# -------------------------
MAX_PARALLEL_JOBS=2            # Conservative for single GPU
GPU_JOBS_LIMIT=1               # Single GPU system
MEMORY_PER_JOB="2G"

# Multi-terminal safety: Generate unique run identifier
RUN_ID="${USER}_$(hostname)_$$_$(date +%s%N | cut -c1-13)"

# Evaluation params (tune)
EPISODES=1
CHECKPOINT="checkpoint_600.pt"
CONTROLLER="OSC_POSE"
RESULT_ROOT="./results_${RUN_ID}"
VIDEO_ROOT="${RESULT_ROOT}/videos"
TEMP_JOBS_DIR="${RESULT_ROOT}/temp_jobs"

# Artifacts (example set - edit as needed)
declare -A ARTIFACT_PATHS=(
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

# Morphs & tasks (example)
MORPHS=("UR5e") #"Panda")
TASKS=("Door" "Lift") #"PickPlaceCan" "Wipe")

# Make directories
mkdir -p "$RESULT_ROOT"
mkdir -p "$VIDEO_ROOT"
mkdir -p "$TEMP_JOBS_DIR"

echo "🔒 Multi-terminal safety enabled. Run ID: $RUN_ID"
echo "⚡ Parallel execution: MAX_JOBS=$MAX_PARALLEL_JOBS, GPU_LIMIT=$GPU_JOBS_LIMIT"
echo "📁 Results root: $RESULT_ROOT"

# Create main aggregated results file
AGGREGATED_RESULTS="$RESULT_ROOT/aggregated_results_${RUN_ID}.csv"
echo "Method,Seed,Morph,Task,Success_Rate,Avg_Reward,Std_Reward" > "$AGGREGATED_RESULTS"

# Helper: extract success / reward from stdout
extract_values() {
    local output="$1"
    local success_rate
    local avg_reward

    success_rate=$(echo "$output" | grep -oP 'Success Rate:\s*\K[0-9.]+')
    avg_reward=$(echo "$output" | grep -oP 'Avg\. Reward:\s*\K[0-9.]+')

    success_rate=${success_rate:-0.0}
    avg_reward=${avg_reward:-0.0}

    echo "${success_rate},${avg_reward}"
}
export -f extract_values

# Worker function called by GNU parallel
run_single_eval() {
    # positional args:
    # 1 job_id 2 key 3 run_dir 4 morph 5 task 6 seed 7 method
    local job_id="$1"
    local key="$2"
    local run_dir="$3"
    local morph="$4"
    local task="$5"
    local seed="$6"
    local method="$7"

    local job_output_file="${TEMP_JOBS_DIR}/job_${job_id}_${method}_${morph}_${task}_seed${seed}.txt"
    local out_dir="${RESULT_ROOT}"

    # GPU device assignment for proper resource management
    local gpu_id=$((job_id % GPU_JOBS_LIMIT))
    export CUDA_VISIBLE_DEVICES=$gpu_id

    echo "[JOB-$job_id] Starting: $method $morph $task (GPU=$gpu_id, seed $seed)" > "$job_output_file"

    # Ensure run_dir exists (best-effort)
    if [[ ! -d "$run_dir" ]]; then
        echo "FAILED: run_dir not found: $run_dir" >> "$job_output_file"
        return 1
    fi

    # Build command
    local CMD=(python tools/zs_morph_eval.py
        --run_dir "$run_dir"
        --checkpoint "$CHECKPOINT"
        --morph "$morph"
        --task "$task"
        --controller "$CONTROLLER"
        --episodes "$EPISODES"
        --save_video "${VIDEO_ROOT}/${key}_${morph}_${task}/"
        --save_metrics
        --out_dir "$out_dir"
        --seed "$seed"
    )

    # Execute and capture stdout/stderr
    local output
    if output=$("${CMD[@]}" 2>&1); then
        # attempt to parse numbers from stdout as quick-check
        local vals
        vals=$(extract_values "$output")
        echo "SUCCESS,${vals}" >> "$job_output_file"
        echo "[JOB-$job_id] ✅ Success results: ${vals}" >> "$job_output_file"
        # Also save full stdout for debugging
        echo "=== STDOUT/STDERR ===" >> "$job_output_file"
        echo "$output" >> "$job_output_file"
    else
        echo "FAILED,0.0,0.0" >> "$job_output_file"
        echo "[JOB-$job_id] ❌ FAILED" >> "$job_output_file"
        echo "=== ERROR OUTPUT ===" >> "$job_output_file"
        echo "$output" >> "$job_output_file"
        return 1
    fi
}
export -f run_single_eval

# Build job list with method extraction
JOB_LIST_FILE="${TEMP_JOBS_DIR}/job_list.txt"
: > "$JOB_LIST_FILE"
job_counter=0

# Get unique method names from artifact keys
declare -A METHODS
for key in "${!ARTIFACT_PATHS[@]}"; do
    method=$(echo "$key" | sed 's/_seed[0-9]*$//')
    METHODS[$method]=1
done

echo "🔄 Generating parallel job list for methods: ${!METHODS[@]}"

# Generate jobs: for each artifact key, morph, task -> with method extraction
for key in "${!ARTIFACT_PATHS[@]}"; do
    run_dir="${ARTIFACT_PATHS[$key]}"
    method=$(echo "$key" | sed 's/_seed[0-9]*$//')
    
    # Try to parse seed from path name (pattern: _1409-run or similar)
    seed_candidate=$(echo "$run_dir" | grep -oP '_\K[0-9]{3,6}(?=-run)' || true)
    if [[ -z "$seed_candidate" ]]; then
        # fallback numeric suffix in key (e.g., _seed1)
        seed_candidate=$(echo "$key" | grep -oP '(?<=_seed)[0-9]+' || true)
        seed_candidate=${seed_candidate:-1}
    fi

    for morph in "${MORPHS[@]}"; do
        for task in "${TASKS[@]}"; do
            job_counter=$((job_counter + 1))
            echo "$job_counter $key $run_dir $morph $task $seed_candidate $method" >> "$JOB_LIST_FILE"
        done
    done
done

echo "📊 Generated $job_counter jobs across ${#METHODS[@]} methods (job list: $JOB_LIST_FILE)"

# Export runtime variables for GNU parallel child processes
export TEMP_JOBS_DIR RESULT_ROOT CHECKPOINT EPISODES CONTROLLER VIDEO_ROOT RUN_ID GPU_JOBS_LIMIT

# Run with GNU parallel
echo "⚡ Launching jobs with GNU parallel (jobs=$MAX_PARALLEL_JOBS)..."
parallel --jobs "$MAX_PARALLEL_JOBS" --colsep ' ' --joblog "${RESULT_ROOT}/parallel_execution_log.txt" --progress \
    --bar --eta --memfree "$MEMORY_PER_JOB" \
    run_single_eval {1} {2} {3} {4} {5} {6} {7} :::: "$JOB_LIST_FILE"

echo "✅ All parallel jobs completed. Aggregating results with statistical analysis..."

# -------------------------
# Enhanced Aggregation + Statistical Analysis
# -------------------------
echo "📊 Processing results by method-morph-task combinations..."

# Process individual job results into aggregated CSV
for method in "${!METHODS[@]}"; do
    echo "🔬 Processing method: $method"
    
    for morph in "${MORPHS[@]}"; do
        for task in "${TASKS[@]}"; do
            # Collect results from all seeds for this combination
            declare -a success_rates=()
            declare -a avg_rewards=()
            declare -a seed_numbers=()
            
            # Find all job files for this method-morph-task combination
            for job_file in "$TEMP_JOBS_DIR"/job_*_${method}_${morph}_${task}_seed*.txt; do
                if [[ -f "$job_file" ]]; then
                    # Extract seed number from filename
                    seed_num=$(echo "$(basename "$job_file")" | grep -oP 'seed\K[0-9]+')
                    
                    # Extract SUCCESS line
                    result_line=$(grep "^SUCCESS," "$job_file" 2>/dev/null || echo "FAILED,0.0,0.0")
                    
                    if [[ "$result_line" == SUCCESS,* ]]; then
                        success_rate=$(echo "$result_line" | cut -d',' -f2)
                        avg_reward=$(echo "$result_line" | cut -d',' -f3)
                        
                        success_rates+=($success_rate)
                        avg_rewards+=($avg_reward)
                        seed_numbers+=($seed_num)
                        
                        # Add individual result to main CSV
                        echo "$method,$seed_num,$morph,$task,$success_rate,$avg_reward,0.0" >> "$AGGREGATED_RESULTS"
                    fi
                fi
            done
            
            # Calculate and display aggregated statistics
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
                
                echo "  ✅ $morph-$task: Success=${success_mean}±${success_std}%, Reward=${reward_mean}±${reward_std} (${#success_rates[@]} seeds: ${seed_numbers[*]})"
            else
                echo "  ❌ $morph-$task: No valid results found"
            fi
            
            # Clear arrays for next iteration
            unset success_rates avg_rewards seed_numbers
            declare -a success_rates=()
            declare -a avg_rewards=()
            declare -a seed_numbers=()
        done
    done
done

# Also run original JSON aggregation for compatibility
python3 - <<'PY'
import os, json, sys, glob
import pandas as pd
import numpy as np

result_root = os.environ.get("RESULT_ROOT", "./results")
out_rows = []

# Search for jsons under result_root/*/*_metrics.json
for root, dirs, files in os.walk(result_root):
    for fname in files:
        if fname.endswith("_metrics.json"):
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r") as f:
                    d = json.load(f)
                # Normalize keys
                row = {
                    "artifact": d.get("artifact", os.path.basename(fpath)),
                    "type": d.get("type", "unknown"),
                    "seed": d.get("seed", None),
                    "run_version": d.get("run_version", None),
                    "morph": d.get("morph", None),
                    "task": d.get("task", None),
                    "episodes": d.get("episodes", None),
                    "avg_reward": float(d.get("avg_reward", 0.0)),
                    "std_reward": float(d.get("std_reward", 0.0)),
                    "success_rate_pct": float(d.get("success_rate_pct", 0.0)),
                    "json_path": fpath
                }
                out_rows.append(row)
            except Exception as e:
                print("Failed to read", fpath, e, file=sys.stderr)

if not out_rows:
    print("No per-run JSON metrics found under", result_root)
    # Still create empty csv for downstream scripts
    pd.DataFrame(columns=["artifact","type","seed","morph","task","avg_reward","std_reward","success_rate_pct"]).to_csv(os.path.join(result_root,"final_results.csv"), index=False)
    sys.exit(0)

df = pd.DataFrame(out_rows)
df.to_csv(os.path.join(result_root,"final_results.csv"), index=False)
print("Saved aggregated per-run CSV to:", os.path.join(result_root,"final_results.csv"))

# Create summary aggregated table by variant (type) and task
grouped = df.groupby(['type','task']).agg({
    'avg_reward': ['mean','std'],
    'success_rate_pct': ['mean','std'],
    'artifact': 'count'
}).round(2)

grouped.columns = ['_'.join(x) for x in grouped.columns]
grouped = grouped.reset_index()
print("\nSummary (type x task):")
print(grouped.to_string(index=False))

# Save summary
grouped.to_csv(os.path.join(result_root,"summary_by_type_task.csv"), index=False)
PY

# Enhanced visualization
export AGGREGATED_RESULTS RESULT_ROOT RUN_ID
echo "📈 Generating enhanced analysis plots..."

python3 - <<'EOF'
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
import sys

try:
    # Read the aggregated results
    aggregated_file = os.environ.get('AGGREGATED_RESULTS', '')
    results_dir = os.environ.get('RESULT_ROOT', './results')
    run_id = os.environ.get('RUN_ID', 'run_unknown')
    
    if not aggregated_file or not os.path.exists(aggregated_file):
        print(f"❌ Cannot find aggregated results file: {aggregated_file}")
        sys.exit(1)
    
    df = pd.read_csv(aggregated_file)
    print(f"\n📊 Loaded aggregated data: {df.shape[0]} rows, {len(df['Method'].unique())} methods")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    methods = df['Method'].unique()
    morphs = df['Morph'].unique()
    tasks = df['Task'].unique()
    
    # Create comprehensive analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Success rate by method and morph-task combination
    if len(methods) > 1:
        success_pivot = df.pivot_table(values='Success_Rate', index='Method', 
                                       columns=['Morph', 'Task'], aggfunc='mean', fill_value=0)
        if not success_pivot.empty:
            sns.heatmap(success_pivot, annot=True, fmt='.1f', ax=axes[0,0], cmap='RdYlGn')
            axes[0,0].set_title('Success Rate by Method and Morph-Task (%)')
            axes[0,0].set_ylabel('Method')
    else:
        # Single method: show by seed
        success_by_seed = df.pivot_table(values='Success_Rate', index='Seed', 
                                         columns=['Morph', 'Task'], aggfunc='mean', fill_value=0)
        if not success_by_seed.empty:
            sns.heatmap(success_by_seed, annot=True, fmt='.1f', ax=axes[0,0], cmap='RdYlGn')
            axes[0,0].set_title('Success Rate by Seed and Morph-Task (%)')
            axes[0,0].set_ylabel('Seed')
    
    # Plot 2: Reward comparison
    if len(methods) > 1:
        reward_pivot = df.pivot_table(values='Avg_Reward', index='Method', 
                                      columns=['Morph', 'Task'], aggfunc='mean', fill_value=0)
        if not reward_pivot.empty:
            sns.heatmap(reward_pivot, annot=True, fmt='.1f', ax=axes[0,1], cmap='viridis')
            axes[0,1].set_title('Average Reward by Method and Morph-Task')
            axes[0,1].set_ylabel('Method')
    else:
        reward_by_seed = df.pivot_table(values='Avg_Reward', index='Seed', 
                                        columns=['Morph', 'Task'], aggfunc='mean', fill_value=0)
        if not reward_by_seed.empty:
            sns.heatmap(reward_by_seed, annot=True, fmt='.1f', ax=axes[0,1], cmap='viridis')
            axes[0,1].set_title('Average Reward by Seed and Morph-Task')
            axes[0,1].set_ylabel('Seed')
    
    # Plot 3: Method comparison (if multiple methods)
    if len(methods) > 1:
        method_stats = df.groupby('Method').agg({'Success_Rate': ['mean', 'std'], 
                                                 'Avg_Reward': ['mean', 'std']}).round(2)
        method_stats.columns = ['_'.join(col) for col in method_stats.columns]
        
        x_pos = np.arange(len(methods))
        axes[1,0].bar(x_pos, method_stats['Success_Rate_mean'], 
                      yerr=method_stats['Success_Rate_std'], capsize=5)
        axes[1,0].set_xlabel('Method')
        axes[1,0].set_ylabel('Success Rate (%)')
        axes[1,0].set_title('Success Rate by Method (Mean ± Std)')
        axes[1,0].set_xticks(x_pos)
        axes[1,0].set_xticklabels(methods, rotation=45)
        axes[1,0].grid(True, alpha=0.3)
        
        axes[1,1].bar(x_pos, method_stats['Avg_Reward_mean'], 
                      yerr=method_stats['Avg_Reward_std'], capsize=5)
        axes[1,1].set_xlabel('Method')
        axes[1,1].set_ylabel('Average Reward')
        axes[1,1].set_title('Average Reward by Method (Mean ± Std)')
        axes[1,1].set_xticks(x_pos)
        axes[1,1].set_xticklabels(methods, rotation=45)
        axes[1,1].grid(True, alpha=0.3)
    else:
        # Single method: show seed variation
        seed_stats = df.groupby(['Morph', 'Task']).agg({'Success_Rate': ['mean', 'std']}).round(2)
        seed_stats.columns = ['_'.join(col) for col in seed_stats.columns]
        seed_stats = seed_stats.reset_index()
        seed_stats['combo'] = seed_stats['Morph'] + '-' + seed_stats['Task']
        
        x_pos = np.arange(len(seed_stats))
        axes[1,0].bar(x_pos, seed_stats['Success_Rate_mean'], 
                      yerr=seed_stats['Success_Rate_std'], capsize=5)
        axes[1,0].set_xlabel('Morph-Task')
        axes[1,0].set_ylabel('Success Rate (%)')
        axes[1,0].set_title('Success Rate by Task (Mean ± Std across seeds)')
        axes[1,0].set_xticks(x_pos)
        axes[1,0].set_xticklabels(seed_stats['combo'], rotation=45)
        axes[1,0].grid(True, alpha=0.3)
        
        # Individual seed performance
        for i, (method, morph, task) in enumerate(df[['Method', 'Morph', 'Task']].drop_duplicates().values):
            method_data = df[(df['Method']==method) & (df['Morph']==morph) & (df['Task']==task)]
            axes[1,1].scatter(method_data['Seed'], method_data['Success_Rate'], 
                             label=f'{morph}-{task}', s=60, alpha=0.7)
        
        axes[1,1].set_xlabel('Seed')
        axes[1,1].set_ylabel('Success Rate (%)')
        axes[1,1].set_title('Success Rate by Seed')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    
    plt.suptitle(f'🚀 Enhanced Zero-Shot Morphology Evaluation (Run: {run_id})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(results_dir, f'enhanced_analysis_{run_id}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Enhanced analysis plot saved to: {plot_path}")
    
    # Print summary statistics
    print("\n📊 Summary Statistics:")
    print("="*80)
    
    if len(methods) > 1:
        for method in methods:
            method_data = df[df['Method'] == method]
            if not method_data.empty:
                mean_success = method_data['Success_Rate'].mean()
                std_success = method_data['Success_Rate'].std()
                mean_reward = method_data['Avg_Reward'].mean()
                std_reward = method_data['Avg_Reward'].std()
                n_runs = len(method_data)
                print(f"{method:15} - Success: {mean_success:.1f}±{std_success:.1f}%, "
                      f"Reward: {mean_reward:.1f}±{std_reward:.1f}, n={n_runs}")
    else:
        # Single method summary
        overall_success = df['Success_Rate'].mean()
        overall_success_std = df['Success_Rate'].std()
        overall_reward = df['Avg_Reward'].mean()
        overall_reward_std = df['Avg_Reward'].std()
        print(f"Overall Performance - Success: {overall_success:.1f}±{overall_success_std:.1f}%, "
              f"Reward: {overall_reward:.1f}±{overall_reward_std:.1f}")
        
        # Per-task breakdown
        for morph in morphs:
            for task in tasks:
                task_data = df[(df['Morph']==morph) & (df['Task']==task)]
                if not task_data.empty:
                    task_success = task_data['Success_Rate'].mean()
                    task_success_std = task_data['Success_Rate'].std()
                    task_reward = task_data['Avg_Reward'].mean()
                    n_seeds = len(task_data)
                    print(f"  {morph}-{task:10} - Success: {task_success:.1f}±{task_success_std:.1f}%, "
                          f"Reward: {task_reward:.1f}, seeds={n_seeds}")
    
    print("="*80)

except Exception as e:
    print(f"❌ Error creating enhanced plots: {e}")
    import traceback
    traceback.print_exc()
EOF

echo "✅ Enhanced aggregation and analysis complete!"

# Try to create simple plots from original format as fallback
python3 - <<'PY2'
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import os
    
    result_root = os.environ.get("RESULT_ROOT", "./results")
    summary_path = os.path.join(result_root, "summary_by_type_task.csv")
    
    if os.path.exists(summary_path):
        grouped = pd.read_csv(summary_path)
        sns.set_style('whitegrid')
        
        # Plot: average success by type/task
        if 'success_rate_pct_mean' in grouped.columns:
            pivot_succ = grouped.pivot(index='type', columns='task', values='success_rate_pct_mean')
            if pivot_succ is not None and not pivot_succ.empty:
                plt.figure(figsize=(10,6))
                pivot_succ.plot(kind='bar', title='Average Success Rate (%) by Type and Task')
                plt.ylabel('Success Rate (%)')
                plt.tight_layout()
                plt.savefig(os.path.join(result_root,"success_by_type_task.png"), dpi=150)
                plt.close()
        
        # Plot: average reward by type/task
        if 'avg_reward_mean' in grouped.columns:
            pivot_reward = grouped.pivot(index='type', columns='task', values='avg_reward_mean')
            if pivot_reward is not None and not pivot_reward.empty:
                plt.figure(figsize=(10,6))
                pivot_reward.plot(kind='bar', title='Average Reward by Type and Task')
                plt.ylabel('Average Reward')
                plt.tight_layout()
                plt.savefig(os.path.join(result_root,"reward_by_type_task.png"), dpi=150)
                plt.close()
        
        print("✅ Created fallback plots from JSON aggregation")
    
except Exception as e:
    print("Fallback plotting failed:", e)
PY2

echo ""
echo "======================================================"
echo "🎉 ENHANCED PARALLEL EVALUATION COMPLETE!"
echo "🔒 Multi-terminal safe Run ID: $RUN_ID"  
echo "⚡ Speedup achieved: ~${MAX_PARALLEL_JOBS}x faster than sequential"
echo "🧠 GPU resource management: Jobs distributed across $GPU_JOBS_LIMIT GPUs"
echo "======================================================"
echo "📊 Results summary:"
echo "  - Enhanced aggregated CSV: ${AGGREGATED_RESULTS}"
echo "  - Original format CSV: ${RESULT_ROOT}/final_results.csv"
echo "  - Summary by type: ${RESULT_ROOT}/summary_by_type_task.csv"
echo "  - Enhanced analysis plot: ${RESULT_ROOT}/enhanced_analysis_${RUN_ID}.png"
echo "  - Individual job logs: ${TEMP_JOBS_DIR}/"
echo "  - Parallel execution log: ${RESULT_ROOT}/parallel_execution_log.txt"
echo "  - Videos: ${VIDEO_ROOT}/"
echo "======================================================"

# Display final aggregated results preview
echo -e "\n📋 Final Aggregated Results Preview:"
echo "============================================"
head -10 "$AGGREGATED_RESULTS" 2>/dev/null || echo "No aggregated results found"

echo -e "\n✨ ANALYSIS COMPLETE!"
echo "🔒 Your unique run identifier: $RUN_ID"
echo "🚀 Ready for multi-terminal parallel usage!"
echo ""
echo "📋 MULTI-TERMINAL USAGE GUIDE:"
echo "  Terminal 1: Uncomment allnodes artifacts and run this script"
echo "  Terminal 2: Uncomment avg_nodes artifacts and run this script" 
echo "  Terminal 3: Uncomment object_only artifacts and run this script"
echo "  Each terminal runs ${MAX_PARALLEL_JOBS} jobs in parallel with unique IDs!"

# Clean up temporary job files (optional)
# echo -e "\n🧹 Cleaning up temporary job files..."
# rm -rf "$TEMP_JOBS_DIR"

echo "DONE."