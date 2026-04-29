import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read CSV file
csv_file = 'mapo_results.csv' # method names
df = pd.read_csv(csv_file)
    # Strip whitespace from column names
df.columns = df.columns.str.strip()

# Define models and languages (order as in the plot)
models = ['W-Reinforce', 'DPO', 'PPO', 'SFT', 'NPO']
languages = ['en', 'de', 'es', 'fr', 'ru']

# Helper to map model string in CSV to display name
def get_model_display_name(model_str):
    if 'w-reinforce' in model_str:
        return 'W-Reinforce'
    elif 'dpo' in model_str:
        return 'DPO'
    elif 'ppo' in model_str:
        return 'PPO'
    elif 'sft' in model_str:
        return 'SFT'
    elif 'npo' in model_str:
        return 'NPO'
    else:
        return model_str

# Prepare bar values: bar_values[lang_idx][model_idx]
bar_values = []
for lang in languages:
    lang_vals = []
    for model in models:
        # Find the row for this language and model
        row = df[(df['source'].str.strip() == lang) & (df['model'].apply(get_model_display_name) == model)]
        if not row.empty:
            val = float(row.iloc[0]['length_controlled_winrate'])
        else:
            val = 0.0
        lang_vals.append(val)
    bar_values.append(lang_vals)

# Reference lines: mean for each language
reference_lines = [np.mean([bar_values[i][j] for j in range(len(models))]) for i in range(len(languages))]

bar_width = 0.15
x = np.arange(len(models))

fig, ax = plt.subplots(figsize=(8, 6))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

for i, (lang, values) in enumerate(zip(languages, bar_values)):
    ax.bar(x + i * bar_width, values, width=bar_width, label=lang, color=colors[i])

# Add horizontal dashed reference lines
for i, ref in enumerate(reference_lines):
    ax.axhline(ref, linestyle='--', color=colors[i], linewidth=2)

ax.set_xticks(x + bar_width * 2)
ax.set_xticklabels(models)
ax.set_xlabel('Model')
ax.set_ylabel('length_controlled_winrate')
ax.set_title('mapo')
ax.legend(title='Language')
plt.tight_layout()
plt.savefig('visualisation/mapo_bar_chart.png')
plt.show()
