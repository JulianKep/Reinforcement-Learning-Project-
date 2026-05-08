# Claude Plan

Phase 1: Environment + random agents
  → Get the game running, make sure rules are correct

Phase 2: PPO against a random opponent
  → Train a basic neural net to beat random play
  → Use stable-baselines3 or CleanRL, don't write PPO from scratch

Phase 3: Naive self-play
  → Play current policy vs itself
  → This already produces interesting bluffing behavior

Phase 4: Opponent pool / league
  → Save checkpoints every N episodes
  → Sample a random old checkpoint as opponent
  → Much more stable than pure self-play

Phase 5 (optional): NFSP
  → If phase 4 results are good, try NFSP for more principled convergence



# Commands
pip freeze > requirements.txt
pip install -r requirements.txt

source env/bin/activate