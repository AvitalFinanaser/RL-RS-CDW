from scripts.run_baselines import run_baseline_evaluation, run_single_baseline
from baselines.baseline_models import *

# Run a single policy
results = run_single_baseline(
    file_path="C:/Users/avita/Desktop/לימודים/תוכנית מיתר/Consenz project/CDW/datasets/event_lists/config001_llm/(CSF=0_events,_APS,_threshold=0.5)/instance_0",
    policy=RandomPolicy(seed=42),
    n_episodes=1,
    seed=123
)
# Run all policies
results = run_baseline_evaluation(
    file_path="C:/Users/avita/Desktop/לימודים/תוכנית מיתר/Consenz project/CDW/datasets/event_lists/config001_llm/(CSF=0_events,_APS,_threshold=0.5)/instance_0",
    n_episodes=1,
    analyze=False,
    csv_name="try_run_script.csv",
    seed=123,
    policies=[RandomPolicy(seed=42)],
)

