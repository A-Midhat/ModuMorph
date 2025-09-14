#!/usr/bin/env python3
"""
Aggregate evaluation results across seeds for paper reporting.
"""
import os
import json
import pandas as pd
import numpy as np
import argparse

def load_all_results(results_root):
    """Load all JSON results from the directory structure."""
    all_results = []
    
    for variant_dir in os.listdir(results_root):
        variant_path = os.path.join(results_root, variant_dir)
        if not os.path.isdir(variant_path):
            continue
            
        variant_type = variant_dir  # allnodes, avg_nodes, object_only
        
        for filename in os.listdir(variant_path):
            if filename.endswith('_metrics.json'):
                json_path = os.path.join(variant_path, filename)
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    
                    seed = extract_seed_from_name(data.get('artifact', filename))
                    
                    result = {
                        'variant': variant_type,
                        'seed': seed,
                        'morph': data.get('morph'),
                        'task': data.get('task'), 
                        'avg_reward': data.get('avg_reward'),
                        'std_reward': data.get('std_reward'),
                        'success_rate_pct': data.get('success_rate_pct'),
                        'episodes': data.get('episodes'),
                        'artifact': data.get('artifact', filename)
                    }
                    all_results.append(result)
                    
                except Exception as e:
                    print(f"Error loading {json_path}: {e}")
    
    return all_results

def extract_seed_from_name(name):
    """Extract seed number from artifact name."""
    import re
    match = re.search(r'_(\d{4})-', name)
    if match:
        return match.group(1)
    return 'unknown'

def aggregate_by_groups(results_df):
    """Aggregate results by (morph, variant, task) groups."""
    grouped = results_df.groupby(['morph', 'variant', 'task'])
    aggregated_results = []
    
    for (morph, variant, task), group in grouped:
        if len(group) == 0:
            continue
            
        mean_reward = group['avg_reward'].mean()
        std_reward = group['avg_reward'].std(ddof=1) if len(group) > 1 else 0
        mean_success = group['success_rate_pct'].mean()
        std_success = group['success_rate_pct'].std(ddof=1) if len(group) > 1 else 0
        n_seeds = len(group)
        seeds_used = sorted(group['seed'].unique())
        
        result = {
            'morph': morph,
            'variant': variant, 
            'task': task,
            'n_seeds': n_seeds,
            'seeds_used': ','.join(map(str, seeds_used)),
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_success_rate': mean_success,
            'std_success_rate': std_success,
            'formatted_reward': f"{mean_reward:.1f} ± {std_reward:.1f}",
            'formatted_success': f"{mean_success:.1f} ± {std_success:.1f}%"
        }
        aggregated_results.append(result)
    
    return pd.DataFrame(aggregated_results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_root', required=True)
    parser.add_argument('--output_dir', required=True) 
    args = parser.parse_args()
    
    print(f"Loading results from: {args.results_root}")
    
    all_results = load_all_results(args.results_root)
    if not all_results:
        print("No results found!")
        return
        
    results_df = pd.DataFrame(all_results)
    print(f"Loaded {len(results_df)} individual evaluation results")
    
    individual_path = os.path.join(args.output_dir, 'individual_results.csv')
    results_df.to_csv(individual_path, index=False)
    print(f"Saved individual results to: {individual_path}")
    
    print("\nData Summary:")
    print(f"Variants: {sorted(results_df['variant'].unique())}")
    print(f"Morphs: {sorted(results_df['morph'].unique())}")
    print(f"Tasks: {sorted(results_df['task'].unique())}")
    print(f"Seeds found: {sorted(results_df['seed'].unique())}")
    
    print("\nAggregating across seeds...")
    aggregated_df = aggregate_by_groups(results_df)
    
    agg_path = os.path.join(args.output_dir, 'final_results.csv')
    aggregated_df.to_csv(agg_path, index=False)
    print(f"Saved final aggregated results to: {agg_path}")
    
    print("\n" + "="*80)
    print("PAPER-READY RESULTS")
    print("="*80)
    
    for morph in sorted(aggregated_df['morph'].unique()):
        morph_data = aggregated_df[aggregated_df['morph'] == morph]
        
        print(f"\n--- {morph.upper()} RESULTS ---")
        print(f"{'Variant':<15} {'Task':<15} {'Reward (Mean±Std)':<20} {'Success Rate (%)':<20} {'N Seeds':<8}")
        print("-" * 80)
        
        for _, row in morph_data.sort_values(['variant', 'task']).iterrows():
            print(f"{row['variant']:<15} {row['task']:<15} {row['formatted_reward']:<20} {row['formatted_success']:<20} {row['n_seeds']:<8}")
    
    print("\n" + "="*80)
    print(f"Results saved to: {agg_path}")

if __name__ == '__main__':
    main()