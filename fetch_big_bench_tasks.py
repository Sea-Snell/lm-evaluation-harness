import os

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
                        task_files[f"{task}_{sub_path}"] = os.path.join(path, sub_path, 'task.json')
        else:
            task_files[task] = os.path.join(path, 'task.json')

    for task_name, task_path in task_files.items():
        os.system(f"cp {task_path} {os.path.join(TARGET_PATH, task_name)}.json")


