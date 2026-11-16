"""Original long recipe, gated by actual new short acceptance and own parent."""
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from final_teacher_config import ROOT, DATA, policies, sha, write_json
from final_teacher_config import OUTPUT as FINAL_OUTPUT
from stage_provenance import parent_checkpoint
from long_refinement_provenance import execution_gate

EXECUTION_GATE = HERE.parent/'long_only_execution_20261115/decision.json'
execution_gate()
PARENT = parent_checkpoint('long')
PARENT_AUDIT = HERE.parent/'final_execution/long_parent_audit.json'
audit = json.loads(PARENT_AUDIT.read_text())
assert audit['checkpoint_sha256'] == PARENT['sha256']
assert audit['actual_all21_optimizer_state_steps'] == PARENT['actual_step']
OUTPUT = FINAL_OUTPUT/'long_reconstruction'
WARMSTART = Path(PARENT['path'])
WARMSTART_SHA256 = PARENT['sha256']
INITIAL_STEP = PARENT['actual_step']
FINETUNING_UPDATES = 2000
FINAL_STEP = INITIAL_STEP+FINETUNING_UPDATES
SAMPLER_SEED = 20261119
LR = 2e-5
POLICIES = {'long': policies()['long']}
