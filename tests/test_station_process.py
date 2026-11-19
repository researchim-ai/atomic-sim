import json
import sys
import subprocess
import pytest
from scripts.station_process import execute,ProcessLimit


def test_success_and_failure_have_separate_operational_receipts(tmp_path):
    r=execute([sys.executable,'-c','print("ready",flush=True)'],cwd=tmp_path,log=tmp_path/'ok.log',timeout_s=3,idle_s=2)
    assert r['completed'] and r['exit_code']==0
    assert json.loads((tmp_path/'ok.run.json').read_text())['scope'].startswith('Operational')
    with pytest.raises(subprocess.CalledProcessError):execute([sys.executable,'-c','raise SystemExit(3)'],cwd=tmp_path,log=tmp_path/'bad.log',timeout_s=3,idle_s=2)
    assert json.loads((tmp_path/'bad.run.json').read_text())['reason']=='nonzero_exit'


def test_idle_process_is_terminated_and_reported(tmp_path):
    with pytest.raises(ProcessLimit,match='progress_timeout'):
        execute([sys.executable,'-c','import time;print("started",flush=True);time.sleep(10)'],cwd=tmp_path,log=tmp_path/'idle.log',timeout_s=2,idle_s=.3)
    r=json.loads((tmp_path/'idle.run.json').read_text());assert not r['completed'] and r['exit_code']!=0


def test_chatty_process_still_has_a_total_wall_limit(tmp_path):
    script='import time\nwhile True:\n print("progress",flush=True)\n time.sleep(.03)'
    with pytest.raises(ProcessLimit,match='wall_timeout'):execute([sys.executable,'-c',script],cwd=tmp_path,log=tmp_path/'wall.log',timeout_s=.5,idle_s=.4)


@pytest.mark.parametrize('args',[dict(timeout_s=True),dict(idle_s=0),dict(timeout_s=float('nan')),dict(timeout_s=1,idle_s=2)])
def test_ambiguous_budgets_are_refused_before_launch(tmp_path,args):
    with pytest.raises(ValueError):execute([sys.executable,'-c','pass'],cwd=tmp_path,log=tmp_path/'bad.log',**args)
    assert not (tmp_path/'bad.log').exists()
