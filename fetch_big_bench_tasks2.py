import os
import json
from collections import defaultdict

BIGBENCH_PATH = '/home/csnell/BIG-bench/'
TARGET_PATH = '/home/csnell/lm-evaluation-harness/lm_eval/datasets/bigbench_resources/'

if __name__ == "__main__":
    task_files = {}
    type_counts = defaultdict(int)
    for task in os.listdir(os.path.join(BIGBENCH_PATH, 'bigbench', 'benchmark_tasks')):
        path = os.path.join(
            BIGBENCH_PATH,
            'bigbench',
            'benchmark_tasks',
            task,
        )
        has_subdirs = False

        if not os.path.isdir(path):
            continue
        
        for sub_path in os.listdir(path):
            if os.path.isdir(os.path.join(path, sub_path)):
                if os.path.exists(os.path.join(path, sub_path, 'task.json')):
                    task_files[f"bb_full-{task}-{sub_path}"] = os.path.join(path, sub_path, 'task.json')
                    has_subdirs = True
        
        if not has_subdirs:
            if os.path.exists(os.path.join(path, 'task.json')):
                task_files[f"bb_full-{task}"] = os.path.join(path, 'task.json')
    
    # total = 0
    # for task_name, task_path in task_files.items():
    #     with open(task_path, 'r') as f:
    #         n = len(json.load(f)['examples'])
    #         print(task_name, n)
    #         total += n
    # print(total)
    # print(type_counts)
    # print(len(task_files))

    for task_name, task_path in task_files.items():
        os.system(f"cp {task_path} {os.path.join(TARGET_PATH, task_name)}.json")


