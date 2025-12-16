import json, collections, sys
fn = r"N:\rlvr\rlvr\passk_out\activity_Qwen2.5-7B-Instruct_predictions_passk.jsonl"
ctr = collections.Counter()
with open(fn, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        ctr[(r["sample_idx"], r["k"])] += 1
dups = sum(c-1 for c in ctr.values() if c>1)
print("Total lines:", sum(ctr.values()))
print("Unique (sample_idx,k):", len(ctr))
print("Duplicates (extra lines):", dups)