#!/usr/bin/env python3
"""
Generate table_methods_comparison.tex with correct data from CSV files.
Extracts data for: SFT, PPO, NPO (Ckpt-5), DPO, W-Reinforce methods.
"""
import csv
from collections import defaultdict
import os

# Configuration
CSV_FILES = {
    'ICR': 'results/full_npo/RAIL CrossLingual Transfer - ICR_Results_Run_01.csv',
    'LaCoMSA': 'results/full_npo/RAIL CrossLingual Transfer - LACOMSA_Results_Run_01.csv',
    'MAPO': 'results/full_npo/RAIL CrossLingual Transfer - MAPO_Results_Run_01.csv',
}

METHODS_TO_EXTRACT = ['sft', 'ppo', 'npo_checkpoint-5', 'dpo', 'w-reinforce']
LANGUAGES = ['es', 'ru', 'en', 'de', 'fr']
ID_LANGUAGES = ['es', 'ru']
OOD_LANGUAGES = ['en', 'de', 'fr']

METHOD_DISPLAY_NAMES = {
    'sft': 'SFT',
    'ppo': 'PPO',
    'npo_checkpoint-5': 'NPO (Ckpt-5)',
    'dpo': 'DPO',
    'w-reinforce': 'W-Reinforce',
}

def extract_method(model_name):
    """Extract the method name from full model name."""
    model_lower = model_name.lower()
    
    if 'w-reinforce' in model_lower:
        # Only match w-reinforce without suffix (not _0.1 or _10)
        if 'w-reinforce_' not in model_lower:
            return 'w-reinforce'
    
    for method in ['sft', 'ppo', 'dpo', 'npo_checkpoint-5']:
        if method in model_lower:
            return method
    
    return None

def process_csv(filepath):
    """Read CSV and extract data for target methods."""
    data = defaultdict(lambda: {})  # method -> {language -> (LC, WR)}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        # Read header and strip whitespace
        header = f.readline().strip()
        # Manually parse the header due to potential spacing issues
        reader = csv.DictReader(f, fieldnames=[col.strip() for col in header.split(',')])
        
        for row in reader:
            if row is None:
                continue
            
            # Strip all keys and values
            row = {k.strip(): v.strip() if v else '' for k, v in row.items()}
            
            language = row.get('source', '').strip()
            model_name = row.get('model', '').strip()
            
            if not language or not model_name:
                continue
            
            method = extract_method(model_name)
            if method not in METHODS_TO_EXTRACT:
                continue
            
            try:
                lc = float(row.get('length_controlled_winrate', 0))
                wr = float(row.get('win_rate', 0))
                data[method][language] = (round(lc, 2), round(wr, 2))
            except (ValueError, TypeError):
                continue
    
    return data

def compute_average(method_data):
    """Compute average LC and WR across all languages."""
    lcs = [v[0] for v in method_data.values() if v]
    wrs = [v[1] for v in method_data.values() if v]
    
    avg_lc = sum(lcs) / len(lcs) if lcs else 0
    avg_wr = sum(wrs) / len(wrs) if wrs else 0
    
    return (round(avg_lc, 2), round(avg_wr, 2))

def generate_latex_table(all_data):
    """Generate LaTeX table content."""
    lines = []
    
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabular}{ll|cc|cc|cc|cc|cc|cc}")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{\textbf{Method}} & \multirow{2}{*}{\textbf{Approach}} & \multicolumn{4}{c|}{\textbf{In-Distribution (ID)}} & \multicolumn{6}{c|}{\textbf{Out-of-Distribution (OOD)}} & \multicolumn{2}{c}{\textbf{Avg}} \\")
    lines.append(r" &  & \multicolumn{2}{c}{\textbf{es}} & \multicolumn{2}{c|}{\textbf{ru}} & \multicolumn{2}{c}{\textbf{en}} & \multicolumn{2}{c}{\textbf{de}} & \multicolumn{2}{c|}{\textbf{fr}} & LC & WR \\")
    lines.append(r" &  & LC & WR & LC & WR & LC & WR & LC & WR & LC & WR &  &  \\")
    lines.append(r"\midrule")
    
    for training_method in ['ICR', 'LaCoMSA', 'MAPO']:
        training_data = all_data[training_method]
        first_row = True
        
        for opt_method in METHODS_TO_EXTRACT:
            if opt_method not in training_data:
                continue
            
            method_data = training_data[opt_method]
            avg_lc, avg_wr = compute_average(method_data)
            
            # Build row
            row_parts = []
            
            if first_row:
                row_parts.append(f"\\multirow{{5}}{{*}}{{\\textbf{{{training_method}}}}}")
                first_row = False
            else:
                row_parts.append(" ")
            
            row_parts.append(METHOD_DISPLAY_NAMES[opt_method])
            
            # Add values for each language
            for lang in LANGUAGES:
                if lang in method_data:
                    lc, wr = method_data[lang]
                    row_parts.append(f"{lc}")
                    row_parts.append(f"{wr}")
                else:
                    row_parts.append("--")
                    row_parts.append("--")
            
            # Add averages
            row_parts.append(f"{avg_lc}")
            row_parts.append(f"{avg_wr}")
            
            lines.append(" & ".join(row_parts) + " \\\\")
        
        lines.append(r"\midrule")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Length-controlled win rate (LC) and win rate (WR) across different training approaches (SFT, PPO, NPO, DPO, W-Reinforce), organized by language distribution.}")
    lines.append(r"\label{tab:methods_comparison}")
    lines.append(r"\end{table*}")
    
    return "\n".join(lines)

def main():
    os.chdir('/Users/maianhpham/Documents/NeMA_result')
    
    # Extract data from all three CSV files
    all_data = {}
    for training_method, csv_file in CSV_FILES.items():
        print(f"Processing {training_method}...")
        data = process_csv(csv_file)
        all_data[training_method] = data
        
        # Print extracted data for verification
        for opt_method in METHODS_TO_EXTRACT:
            if opt_method in data:
                print(f"  {opt_method}: {data[opt_method]}")
    
    # Generate LaTeX table
    latex_content = generate_latex_table(all_data)
    
    # Write to file
    with open('table_methods_comparison.tex', 'w') as f:
        f.write(latex_content)
    
    print("\nGenerated table_methods_comparison.tex")

if __name__ == '__main__':
    main()
