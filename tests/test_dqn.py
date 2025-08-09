# tests/test_dqn.py
from environment.env import CollaborativeDocRecEnv
from environment.loaders import ParagraphsLoader, AgentsLoader
from rl_algorithms.dqn_runner import DQNRunner


def main():
    # Loaders setup (adjust paths)
    paragraph_loader = ParagraphsLoader("datasets/event_lists/config001_llm/.../instance_0")
    agents_loader = AgentsLoader("datasets/event_lists/config001_llm/.../instance_0")

    # Environment
    env = CollaborativeDocRecEnv(
        paragraphs_loader=paragraph_loader,
        agents_loader=agents_loader,
        render_mode="human"
    )

    # DQN Runner
    dqn_runner = DQNRunner(env, seed=42)

    # Training
    print("Training DQN...")
    dqn_runner.train(num_episodes=100)

    # Evaluation
    print("Evaluating DQN...")
    results = dqn_runner.evaluate(num_episodes=10)
    print("DQN Results:", results)

    # Save trained model
    dqn_runner.save("results/dqn_model.zip")


if __name__ == "__main__":
    main()
