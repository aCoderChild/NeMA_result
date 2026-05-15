#!/usr/bin/env python3
"""
Script to compute average win rate and length-controlled win rate scores 
across languages for different training methods.

Reads results from ICR, LaCoMSA, and MAPO experiments and outputs aggregated metrics.
"""

import csv
import pandas as pd
from pathlib import Path
from collections import defaultdict

def extract_method(model_name):
    """Extract the method name from the full model name."""
    import re
    
    # Find the training method
    methods = ['sft', 'ppo', 'dpo', 'w-reinforce']
    for method in methods:
        if method in model_name:
            # Handle w-reinforce with parameters
            if method == 'w-reinforce':
                if 'w-reinforce_' in model_name:
                    # Extract parameter like w-reinforce_0.1 or w-reinforce_10
                    match = re.search(r'w-reinforce_([0-9.]+)', model_name)
                    if match:
                        param = match.group(1)
                        return f"w-reinforce_{param}"
                    return 'w-reinforce'
                else:
                    return 'w-reinforce'
            return method
    
    # Handle npo checkpoints - use regex to match exact checkpoint numbers
    match = re.search(r'npo_checkpoint-(\d+)', model_name)
    if match:
        checkpoint_num = match.group(1)
        return f'npo_checkpoint-{checkpoint_num}'
    
    # Handle plain npo
    if 'npo' in model_name:
        return 'npo'
    
    return None

def process_results_file(filepath, training_method_name):
    """Process a single results CSV file."""
    results = defaultdict(list)
    
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Strip whitespace from keys
                row = {k.strip(): v for k, v in row.items()}
                
                model = row.get('model', '').strip()
                lang = row.get('source', '').strip()
                lc = float(row.get('length_controlled_winrate', 0))
                wr = float(row.get('win_rate', 0))
                
                method = extract_method(model)
                if method:
                    results[method].append({
                        'language': lang,
                        'lc': lc,
                        'wr': wr,
                        'training_method': training_method_name
                    })
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
    
    return results

def aggregate_results(all_results):
    """Aggregate results across languages for each method."""
    aggregated = []
    
    for training_method in ['ICR', 'LaCoMSA', 'MAPO']:
        if training_method not in all_results:
            continue
        
        for method, data_list in all_results[training_method].items():
            if not data_list:
                continue
            
            # Calculate averages across languages
            avg_lc = sum(d['lc'] for d in data_list) / len(data_list)
            avg_wr = sum(d['wr'] for d in data_list) / len(data_list)
            num_languages = len(set(d['language'] for d in data_list))
            
            aggregated.append({
                'Training_Method': training_method,
                'Optimization_Method': method,
                'Avg_LC': round(avg_lc, 2),
                'Avg_WR': round(avg_wr, 2),
                'Num_Languages': num_languages,
                'Num_Samples': len(data_list)
            })
    
    return aggregated

def main():
    # Define input files
    input_files = {
        'ICR': 'results/full_npo/RAIL CrossLingual Transfer - ICR_Results_Run_01.csv',
        'LaCoMSA': 'results/full_npo/RAIL CrossLingual Transfer - LACOMSA_Results_Run_01.csv',
        'MAPO': 'results/full_npo/RAIL CrossLingual Transfer - MAPO_Results_Run_01.csv'
    }
    
    # Output file
    output_file = 'analysis/methods_average_metrics.csv'
    
    # Create output directory if it doesn't exist
    Path('analysis').mkdir(exist_ok=True)
    
    all_results = {}
    
    # Process each file
    for training_method, filepath in input_files.items():
        print(f"Processing {training_method}: {filepath}")
        results = process_results_file(filepath, training_method)  # training_method is passed as training_method_name parameter
        all_results[training_method] = results
        print(f"  Found {len(results)} unique methods")
    
    # Aggregate results
    aggregated = aggregate_results(all_results)
    
    # Sort by training method, then by optimization method
    aggregated = sorted(aggregated, key=lambda x: (x['Training_Method'], x['Optimization_Method']))
    
    # Save to CSV
    df = pd.DataFrame(aggregated)
    df.to_csv(output_file, index=False)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Total records: {len(aggregated)}")
    print("\nSummary:")
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()
