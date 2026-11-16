"""Original long budget recipe bound to the actual new long FT selection."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from final_teacher_config import ROOT, DATA, policies, sha, write_json
from final_teacher_config import OUTPUT as FINAL_OUTPUT
from long_refinement_provenance import execution_gate, selected_long_reconstruction_checkpoint

EXECUTION_GATE = HERE.parent/'long_only_execution_20261115/decision.json'
execution_gate()
PARENT = selected_long_reconstruction_checkpoint()
OUTPUT = FINAL_OUTPUT/'long_budget_features'
WARMSTART = Path(PARENT['path'])
WARMSTART_SHA256 = PARENT['sha256']
PARENT_STEP = PARENT['actual_step']
PARENT_LOCAL_FT_UPDATES = PARENT['new_updates']
SOURCE_STEPS = dict(spatial=PARENT_STEP, budget=PARENT_STEP)
UPDATES = 2000
SAMPLER_SEED = 20261120
FEATURE_LR, LR = 1e-6, 1e-6
POLICIES = {'long': policies()['long']}
