"""User-authorized sharing of GPUs0--4, based on actual free memory."""
import json
from pathlib import Path
import subprocess
import time

ALLOWED = None  # User explicitly permits every card with enough free memory.
EVAL_FREE_MIB = 12 * 1024
TRAIN_FREE_MIB = 15 * 1024


def snapshot():
    raw = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=30)
    rows = []
    for line in raw.splitlines():
        i, uuid, total, used, utilization = [v.strip() for v in line.split(',')]
        if ALLOWED is None or i in ALLOWED:
            rows.append(dict(index=i, uuid=uuid, total_MiB=int(total), used_MiB=int(used),
                             free_MiB=int(total)-int(used), utilization=int(utilization)))
    return sorted(rows, key=lambda row: (-row['free_MiB'], int(row['index'])))


def eligible(min_free, exclude=(), idle=False):
    return [row for row in snapshot() if row['index'] not in exclude
            and row['free_MiB'] >= min_free and (not idle or row['utilization'] <= 5)]


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def wait_training(state, state_path):
    while True:
        available = eligible(TRAIN_FREE_MIB)
        if len(available) >= 4:
            return available[:4]
        state.update(status='waiting_for_four_training_GPUs_with_15GiB_free',
                     GPU_snapshot=snapshot(), last_checked=time.time())
        write(state_path, state)
        time.sleep(20)
