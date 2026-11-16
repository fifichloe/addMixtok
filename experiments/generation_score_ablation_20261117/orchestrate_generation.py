"""Evaluate the three frozen-generator ablations as soon as U stages finish."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
KINDS = ('without_gain', 'without_white_all_explicit', 'without_cam_all_explicit')
STATUS = HERE / 'generation_orchestrator_status.json'


def write(**values):
    values['updated_at_unix'] = time.time()
    temporary = STATUS.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(values, indent=2) + '\n')
    temporary.replace(STATUS)
    print(json.dumps(values), flush=True)


def wait_file(path, kind):
    while not path.exists():
        write(status='waiting_for_U_training', kind=kind, expected=str(path))
        time.sleep(60)


def main():
    with (HERE / 'generation_orchestrator.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for kind in KINDS:
                report = HERE / f'budget_{kind}/report.json'
                wait_file(report, kind)
                saved = json.loads(report.read_text())
                assert saved['status'] == 'completed' and saved['new_U_updates'] == 2000
                output = HERE / f'generation_{kind}_seed4_50k'
                command = [sys.executable, str(HERE / 'evaluate.py'),
                    '--kind', 'long', '--ablation', kind,
                    '--cfg', '3.0', '--stage2-steps', '8', '--seed', '4',
                    '--output', str(output)]
                logpath = HERE / f'generation_{kind}_launcher.log'
                with logpath.open('x') as log:
                    child = subprocess.Popen(command, cwd=HERE.parent.parent,
                                             stdout=log, stderr=subprocess.STDOUT)
                    write(status='generation_running', kind=kind, pid=child.pid,
                          output=str(output), log=str(logpath))
                    code = child.wait()
                if code:
                    raise RuntimeError(f'{kind} generation evaluation failed: {code}; see {logpath}')
                summary = json.loads((output / 'summary.json').read_text())
                assert summary['status'] == 'complete' and summary['n'] == 50000
                assert summary['ablation'] == kind and summary['generation_rounds'] == '8+8'
                write(status='generation_complete', kind=kind,
                      gFID=summary['metrics']['full_refine']['gFID'],
                      mean_generated_2d=summary['tokens']['mean_generated_2d'],
                      summary=str(output / 'summary.json'))
            write(status='complete', all_kinds=list(KINDS), MaskGIT_retrained=False)
        except BaseException as error:
            write(status='failed_no_retry', error=repr(error),
                  traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
