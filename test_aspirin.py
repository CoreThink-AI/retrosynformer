"""
Quick smoke-test: run retrosynthesis on Aspirin with random (untrained) weights.
Results are not chemically meaningful without a trained model.pth.
"""
import yaml
from pathlib import Path

from retrosynformer.runner import init_model
from retrosynformer.inference import RoutePredictor

REPO = Path(__file__).parent
ASPIRIN_SMILES = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"  # Ibuprofen (Aspirin is itself a building block)

config = yaml.safe_load((REPO / "results/config.yaml").read_text())

# Override absolute paths to local data directory
config["context"]["building_blocks"] = str(REPO / "data/standard_building_blocks.csv")
config["context"]["templates_path"] = str(REPO / "data/standard_reaction_templates.pickle")
config["evaluation"]["beam_width"] = 3
config["evaluation"]["max_depth"] = 5

print(f"Target: Ibuprofen (Aspirin is a building block — trivially solved)  SMILES={ASPIRIN_SMILES}")
print("Initializing model (random weights — no checkpoint available)...")
model = init_model(config)

predictor = RoutePredictor(model, config, beam_width=3)
print("Running beam search (beam_width=3)...")
best_beam = predictor.predict_route(ASPIRIN_SMILES, beam_width=3)

if best_beam is None:
    print("No route found.")
elif best_beam.route_solved:
    print(f"Route SOLVED in {len(best_beam.reaction_list)} step(s):")
    for i, rxn in enumerate(best_beam.reaction_list, 1):
        print(f"  Step {i}: {rxn}")
    print(f"Trajectory probability: {best_beam.trajectory_prob:.4e}")
else:
    print(f"Route not solved (unsatisfied frontier after {len(best_beam.reaction_list)} step(s)).")
    print("Reaction steps attempted:")
    for i, rxn in enumerate(best_beam.reaction_list, 1):
        print(f"  Step {i}: {rxn}")
    print(f"Remaining frontier: {best_beam.env.state[-1]}")
