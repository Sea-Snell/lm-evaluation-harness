import os
import json
from collections import defaultdict

TASKS = [
    ('unit_conversion', True),
    ('linguistic_mappings', True),
    ('qa_wikidata', False),
    ('mult_data_wrangling', True),
]

BIGBENCH_PATH = '/home/csnell/BIG-bench/'
TARGET_PATH = '/home/csnell/lm-evaluation-harness/lm_eval/datasets/bigbench_resources/'

if __name__ == "__main__":
    task_files = {}
    type_counts = defaultdict(int)
    for task, has_subdirs in TASKS:
        path = os.path.join(
            BIGBENCH_PATH,
            'bigbench',
            'benchmark_tasks',
            task,
        )
        if has_subdirs:
            for sub_path in os.listdir(path):
                if os.path.isdir(os.path.join(path, sub_path)):
                    if os.path.exists(os.path.join(path, sub_path, 'task.json')):
                        task_files[f"bb_data_study_{task}_{sub_path}"] = os.path.join(path, sub_path, 'task.json')
                        type_counts[task] += 1
        else:
            task_files[f"bb_data_study_{task}"] = os.path.join(path, 'task.json')
            type_counts[task] += 1
    
    total = 0
    for task_name, task_path in task_files.items():
        with open(task_path, 'r') as f:
            n = len(json.load(f)['examples'])
            print(task_name, n)
            total += n
    print(total)
    print(type_counts)

    for task_name, task_path in task_files.items():
        os.system(f"cp {task_path} {os.path.join(TARGET_PATH, task_name)}.json")


