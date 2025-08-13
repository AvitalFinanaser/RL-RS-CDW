# RL-RS-CDW
RL-Guided Recommender for Collaborative Document Writing with Partial Preferences


## Baseline Evaluation Script for Collaborative Writing Recommender
This script evaluates baseline recommendation policies in CDW environments, saving results as CSVs and generating plots for analysis.

### Configuration

- **Input Path**: Specify the dataset of initial state (agents.jason, paragraphs.json, events.json) path using `--file_path`.
- **Policies**: Supports `random`, `popularity`, `cf` (collaborative filtering), or `all` (default).
- **Episodes**: Number of episodes (default: 20).
- **Max Steps**: Maximum steps per episode (default: auto-computed as agents * paragraphs).
- **Seed**: Random seed for reproducibility (default: 42).
- **Output Path**: Results saved to `./results` as CSVs and plots.

### Output

- **CSV Files**:
  - `baseline_results.csv`: Summary of policy performance.
  - `detailed_baseline_results.csv`: Episode-level metrics (if `--analyze` is used).
- **Plots**: Visualizations of total reward, episode length, and completion rate (if `--analyze` is used).
- **Report**: Text summary saved as `baseline_comparison_report.txt`.

### Usage

#### Command-Line Interface

Run from your project directory:

```bash
python run_baselines.py --file_path "/absolute/path/to/dataset/instance_folder" [--policy POLICY] [--episodes N] [--max_steps M] [--analyze] [--seed S]
```

#### Import as a Package

```python
from scripts.run_baselines import run_baseline_evaluation, run_single_baseline
from baselines.baseline_models import *

# Run all policies
results = run_baseline_evaluation(
    file_path="/path/to/dataset",
    n_episodes=20,
    analyze=True
)

# Run a single policy
from baselines.baseline_models import RandomPolicy
results = run_single_baseline(
    file_path="/path/to/dataset",
    policy=RandomPolicy(seed=42),
    n_episodes=1
)
```

#### Example

```bash
python run_baselines.py --file_path "/data/config001_llm/instance_0" --policy random --episodes 1 --seed 123
```
