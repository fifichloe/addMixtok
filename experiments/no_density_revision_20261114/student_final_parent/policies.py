"""Final-design-bound teachers with the unchanged original parent recipe."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from final_teacher_config import policies, sha, write_json, DATA
POLICIES = policies()
