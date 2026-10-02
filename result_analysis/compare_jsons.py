import json



def zip_data(data:dict[list|dict]):
    if all(isinstance(v, list) for v in data.values()):
        r_list = []
        for K, v_list in data.items():
            for r_elem, v_elem in zip(r_list, v_list):
                r_elem[K] = v_elem
            r_list.extend([{K: v_elem} for v_elem in v_list[len(r_list):]])
        return [zip_data(x) for x in r_list]
    if all(isinstance(v, dict) for v in data.values()):
        r_dict = {}
        for K, v_dict in data.items():
            for k, v in v_dict.items():
                if k not in r_dict:
                    r_dict[k] = {}
                r_dict[k][K] = v
        return {k: zip_data(v) for k, v in r_dict.items()}
    if all(isinstance(v, (int, float, str)) for v in data.values()):
        if len(set(data.values())) == 1:
            return list(data.values())[0]
        return data
    return data

data = {}
for K, fn in [
    ("og", "adresso_student/outputs/results_jul30/all_results.json"),
    ("fb", "adresso_student/outputs/results/all_results.json")
]:
    with open(fn, "r") as f:
        data[K] = json.load(f) 


with open("tmp/combined.json", "w") as f:
    json.dump(zip_data(data), f, indent=4)