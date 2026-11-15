"""Bounded JSONL agent protocol. No privileged state or branch methods."""
import json
import sys
from .model import EnvConfig,StationEnv,ActionError

MAX_LINE=65536


def dispatch(env,request):
    if type(request) is not dict or type(request.get('method')) is not str:
        raise ActionError('invalid_request','Request object with method required')
    method=request['method']
    fields={'reset':{'method','seed'},'step':{'method','action'},'observe':{'method'},'actions':{'method'}}
    if method not in fields or not set(request)<=fields[method]:
        raise ActionError('invalid_request','Unknown method or fields')
    if method=='reset':
        observation,info=env.reset(seed=request.get('seed'))
        return dict(observation=observation,info=info)
    if method=='observe':
        observation,info=env.observation();return dict(observation=observation,info=info)
    if method=='actions':return env.available_actions()
    obs,reward,terminated,truncated,info=env.step(request.get('action'))
    return dict(observation=obs,reward=reward,terminated=terminated,truncated=truncated,info=info)


def serve(input_stream,output_stream,env=None):
    env=env if env is not None else StationEnv()
    while True:
        line=input_stream.readline(MAX_LINE+1)
        if not line:break
        oversized=len(line)>MAX_LINE
        if oversized:
            while line and not line.endswith('\n'):line=input_stream.readline(MAX_LINE+1)
            response=dict(ok=False,error=dict(code='request_too_large',reason='JSONL line exceeds 65536 characters'))
        else:
            try:response=dict(ok=True,result=dispatch(env,json.loads(line)))
            except ActionError as error:response=dict(ok=False,error=dict(code=error.code,reason=error.reason))
            except (ValueError,TypeError,OverflowError,RecursionError):response=dict(ok=False,error=dict(code='invalid_request',reason='Invalid JSON request'))
        output_stream.write(json.dumps(response,sort_keys=True,allow_nan=False)+'\n');output_stream.flush()


if __name__=='__main__':serve(sys.stdin,sys.stdout)
