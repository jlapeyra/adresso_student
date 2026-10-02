import re
import json
import pandas as pd


params = {}
all_results = []
with open('adresso_student/logs/slurm/adresso_student_2676931.txt', 'r', encoding='utf-8') as f:
    for line in f:
        for param, key in [
            ('ablation', 'ABALATION'),
            ('link_features', 'LINK FEATURES'),
            ('text_encoder', 'Text_encoder'),
            ('audio_encoder', 'audio_encoder'),
        ]:
            line = line.strip()
            if line.startswith(key + ':'):
                params[param] = line.removeprefix(key+':').strip()
        if m := re.match(r'\s*(\w+)\s*TEST:((?:\s+\w+=[\d.e+-]+)*).*', line):
            exp_name, results_str = m.groups()
            if m2 := re.match(r'(\w+)_fold(\d)+', exp_name):
                ablation, fold = m2.groups()
            else:
                ablation, fold = exp_name, None
            if ablation != params.get('ablation'):
                print(f"Warning: ablation mismatch: {ablation} != {params.get('ablation')}")
            params['ablation'] = ablation
            params['fold'] = fold
            results = {}
            for k, v in (x.split('=') for x in results_str.split()):
                results[k] = float(v)

            all_results.append(params | results)
df = pd.DataFrame(all_results)
df.to_csv("result_analysis/all_results2.csv", index=False)
with open("result_analysis/all_results2.json", "w") as f:
    json.dump(df.to_dict(orient="records"), f, indent=2, default=str)