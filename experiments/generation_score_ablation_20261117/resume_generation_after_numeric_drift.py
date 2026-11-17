"""Finish the two pending GPU evaluations without redoing completed shards."""
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from orchestrate_generation import calibrate

KINDS = ('without_white_all_explicit', 'without_cam_all_explicit')
STATUS = HERE / 'generation_resume_status.json'


def write(**values):
    values['updated_at_unix'] = time.time()
    temporary = STATUS.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(values, indent=2) + '\n')
    temporary.replace(STATUS)
    print(json.dumps(values), flush=True)


def main():
    try:
        for kind in KINDS:
            report = HERE / f'budget_{kind}/report.json'
            while not report.exists():
                write(status='waiting_for_budget_report', kind=kind)
                time.sleep(30)
            assert json.loads(report.read_text())['status'] == 'completed'
            policy = calibrate(kind)
            out = HERE / f'generation_{kind}_seed4_50k'
            if out.exists():
                assert (out / 'summary.json').exists(), out
                write(status='already_complete', kind=kind)
                continue
            command = [sys.executable, str(HERE / 'evaluate.py'), '--kind', 'long',
                       '--ablation', kind, '--cfg', '3.0', '--stage2-steps', '8',
                       '--seed', '4', '--output', str(out), '--policy-file', str(policy)]
            logpath = HERE / f'generation_{kind}_launcher.log'
            with logpath.open('x') as log:
                child = subprocess.Popen(command, cwd=HERE.parent.parent,
                                         stdout=log, stderr=subprocess.STDOUT)
                write(status='generation_running', kind=kind, pid=child.pid)
                code = child.wait()
            if code:
                state = json.loads((out / 'status.json').read_text())
                assert state['completed'] == 50000 and state['status'] == 'failed_preserved_no_retry'
                assert state['error'] == 'AssertionError()'
                assert 'base_shard' in state['traceback']
                write(status='shards_complete_awaiting_numeric_audit', kind=kind)
            else:
                summary = json.loads((out / 'summary.json').read_text())
                assert summary['status'] == 'complete' and summary['n'] == 50000
                write(status='generation_complete', kind=kind,
                      gFID=summary['metrics']['full_refine']['gFID'])
        write(status='complete', all_kinds=list(KINDS))
    except BaseException as error:
        write(status='failed_no_retry', error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
