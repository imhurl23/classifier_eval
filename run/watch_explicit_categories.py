"""Keep a progress receipt and rebuild the final report when every arm completes."""
from datetime import datetime, timezone
import json
import subprocess
import time

from run.explicit_categories import ARMS, OUT, ROOT


def line_count(path):
    if not path.exists():
        return 0
    with path.open('rb') as handle:
        return sum(block.count(b'\n') for block in iter(lambda: handle.read(1024 * 1024), b''))


def main():
    started = time.monotonic()
    last_build = 0
    while time.monotonic() - started < 8000:
        counts = {a: line_count(OUT / a / 'results.jsonl') for a in ARMS}
        inflight = {a: len(list((OUT / a / 'inflight').glob('*'))) for a in ARMS}
        complete = all(n == 3362 and inflight[a] == 0 for a, n in counts.items())
        state = {'updated': datetime.now(timezone.utc).isoformat(), 'counts': counts,
                 'inflight': inflight, 'complete_arms': sum(n == 3362 for n in counts.values()),
                 'status': 'ready_for_analysis' if complete else 'running'}
        if complete or time.monotonic() - last_build > 300:
            command = [str(ROOT / '.venv/bin/python'), '-m', 'analysis.explicit_categories']
            if not complete:
                command.append('--allow-partial')
            with (OUT / 'logs' / 'analysis.log').open('a') as handle:
                result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            state['analysis_exit_code'] = result.returncode
            if complete:
                state['status'] = 'completed' if result.returncode == 0 else 'analysis_failed'
            last_build = time.monotonic()
        (OUT / 'watcher.json').write_text(json.dumps(state, indent=2) + '\n')
        print(json.dumps({'complete_arms': state['complete_arms'], 'status': state['status'], 'counts': counts}), flush=True)
        if complete:
            return
        time.sleep(30)
    raise TimeoutError('Watcher wall-time bound reached; per-arm lifecycle supervisors retain their own shutdown bounds')


if __name__ == '__main__':
    main()
