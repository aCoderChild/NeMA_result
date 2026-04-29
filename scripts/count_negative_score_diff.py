import json

count = 0
with open("lacomsa/train.jsonl", "r") as f:
    for line in f:
        try:
            obj = json.loads(line)
            if obj.get("score_diff", 0) < 0:
                count += 1
        except Exception:
            continue
print(f"Number of pairs with negative score_diff: {count}")
