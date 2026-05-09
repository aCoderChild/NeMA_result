import json
from pathlib import Path


def main():
    jsonl_path = Path("responses/mapo/train.jsonl")
    
    chosen_lengths = []
    rejected_lengths = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            if "chosen" in record:
                chosen_lengths.append(len(record["chosen"]))
            if "rejected" in record:
                rejected_lengths.append(len(record["rejected"]))
    
    avg_chosen = sum(chosen_lengths) / len(chosen_lengths) if chosen_lengths else 0
    avg_rejected = sum(rejected_lengths) / len(rejected_lengths) if rejected_lengths else 0
    
    print(f"Average length of 'chosen': {avg_chosen:.2f}")
    print(f"Average length of 'rejected': {avg_rejected:.2f}")


if __name__ == "__main__":
	main()