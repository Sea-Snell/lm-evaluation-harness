import os
import json
from collections import defaultdict

# TASKS = [
#     'auto_debugging',
#     'bbq_lite_json',
#     'code_line_description',
#     'conceptual_combinations',
#     'conlang_translation',
#     'emoji_movie',
#     'formal_fallacies_syllogisms_negation',
#     'hindu_knowledge',
#     'known_unknowns',
#     'language_identification',
#     'linguistics_puzzles',
#     'logic_grid_puzzle',
#     'logical_deduction',
#     'misconceptions_russian',
#     'novel_concepts',
#     'operators',
#     'parsinlu_reading_comprehension',
#     'play_dialog_same_or_different',
#     'repeat_copy_logic',
#     'strange_stories',
#     'strategyqa',
#     'symbol_interpretation',
#     'vitaminc_fact_verification',
#     'winowhy',
# ]

TASKS = [
    'unit_conversion',
    'linguistic_mappings',
    'qa_wikidata',
    'mult_data_wrangling',
]

# TASKS = [
#     'causal_judgment',
#     'date_understanding',
#     'disambiguation_qa',
#     'dyck_languages',
#     'formal_fallacies_syllogisms_negation',
#     'geometric_shapes',
#     'hyperbaton',
#     'logical_deduction',
#     'movie_recommendation',
#     'navigate',
#     'reasoning_about_colored_objects',
#     'ruin_names',
#     'salient_translation_error_detection',
#     'snarks',
#     'sports_understanding',
#     'temporal_sequences',
#     'tracking_shuffled_objects',
# ]

BIGBENCH_PATH = '/home/csnell/BIG-bench/'
TARGET_PATH = '/home/csnell/lm-evaluation-harness/lm_eval/datasets/bigbench_resources/'

if __name__ == "__main__":
    task_files = {}
    for task in TASKS:
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
                    task_files[f"bb_lite_{task}_{sub_path}"] = os.path.join(path, sub_path, 'task.json')
                    has_subdirs = True
        
        if not has_subdirs:
            if os.path.exists(os.path.join(path, 'task.json')):
                task_files[f"bb_lite_{task}"] = os.path.join(path, 'task.json')
    
    task_counts = dict()
    has_multiple_choice = dict()
    for k, v in task_files.items():
        with open(v, 'r') as f:
            task_data = json.load(f)
            task_counts[k[len('bb_lite_'):]] = len(task_data['examples'])
            has_multiple_choice[k[len('bb_lite_'):]] = 'multiple_choice_grade' in task_data['metrics']
    
    import IPython; IPython.embed(); exit(1)

    # for task_name, task_path in task_files.items():
    #     os.system(f"cp {task_path} {os.path.join(TARGET_PATH, task_name)}.json")


