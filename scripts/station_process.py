"""Bounded local scientific processes; operational receipts stay outside science."""
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from atomic_model_training.spec import atomic_json


class ProcessLimit(TimeoutError):
    def __init__(self,reason,log):
        self.reason=reason;self.log=str(log)
        super().__init__('Scientific process '+reason+'; diagnostic log: '+str(log))


def execute(command,*,cwd,log,timeout_s=7200.,idle_s=600.):
    if type(command) not in (list,tuple) or not command or any(type(x) is not str or not x or '\0' in x for x in command):raise ValueError('Direct process argument vector required')
    for value in (timeout_s,idle_s):
        if type(value) not in (int,float) or not math.isfinite(value) or not .1<=value<=14400:raise ValueError('Process time budget')
    if idle_s>timeout_s:raise ValueError('Idle budget exceeds total budget')
    log=Path(log);cwd=Path(cwd)
    if not cwd.is_dir() or any(p.is_symlink() for p in (log,*log.parents)):raise ValueError('Process work/log path')
    receipt=log.with_suffix('.run.json')
    if receipt.exists():raise ValueError('Existing process receipt refused')
    start=time.monotonic();last_progress=start;size=0;reason=None
    with log.open('x') as stream:
        process=subprocess.Popen(list(command),cwd=cwd,env=dict(os.environ),stdout=stream,stderr=subprocess.STDOUT,start_new_session=os.name!='nt')
        try:
            while process.poll() is None:
                now=time.monotonic();current=log.stat().st_size
                if current!=size:last_progress=now;size=current
                if now-start>timeout_s:reason='wall_timeout';break
                if now-last_progress>idle_s:reason='progress_timeout';break
                try:process.wait(timeout=.1)
                except subprocess.TimeoutExpired:pass
        finally:
            if process.poll() is None:
                try:
                    if os.name!='nt':os.killpg(process.pid,signal.SIGTERM)
                    else:process.terminate()
                except ProcessLookupError:pass
                try:process.wait(timeout=5.)
                except subprocess.TimeoutExpired:
                    try:
                        if os.name!='nt':os.killpg(process.pid,signal.SIGKILL)
                        else:process.kill()
                    except ProcessLookupError:pass
                    process.wait()
        result=dict(scope='Operational metadata only; not scientific JSON',completed=reason is None and process.returncode==0,
            reason=reason or ('completed' if process.returncode==0 else 'nonzero_exit'),exit_code=process.returncode,
            wall_s=time.monotonic()-start,pid=process.pid,timeout_s=timeout_s,idle_s=idle_s)
        atomic_json(receipt,result)
    if reason:raise ProcessLimit(reason,log)
    if process.returncode:raise subprocess.CalledProcessError(process.returncode,list(command))
    return result
