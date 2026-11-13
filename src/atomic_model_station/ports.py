"""Typed station port declarations; connectivity checks do not solve a network."""
from dataclasses import dataclass

UNITS={'thermal':('K','W'),'mechanical':('rad/s','N*m'),
       'electrical_power':('1','W'),'fluid':('Pa','kg/s')}

@dataclass(frozen=True)
class Port:
    owner:str
    name:str
    kind:str
    effort_unit:str
    flow_unit:str
    def __post_init__(self):
        if any(type(s) is not str or not s or len(s)>80 for s in (self.owner,self.name,self.kind,self.effort_unit,self.flow_unit)):
            raise ValueError('Bounded nonempty port strings required')
        if self.kind not in UNITS or (self.effort_unit,self.flow_unit)!=UNITS[self.kind]:
            raise ValueError('Port kind/unit mismatch')


def validate_connections(ports,connections):
    if not isinstance(ports,(tuple,list)) or not 1<=len(ports)<=256:raise ValueError('Port budget1..256')
    if not isinstance(connections,(tuple,list)) or len(connections)>128:raise ValueError('Bounded connection list required')
    by={}
    for p in ports:
        if type(p) is not Port or (p.owner,p.name) in by:raise ValueError('Unique owned Port declarations required')
        by[(p.owner,p.name)]=p
    used=set()
    for pair in connections:
        if not isinstance(pair,(tuple,list)) or len(pair)!=2:raise ValueError('Two endpoints required')
        if any(not isinstance(x,(tuple,list)) or len(x)!=2 or any(type(s) is not str for s in x) for x in pair):
            raise ValueError('Owner/name endpoint pair required')
        a,b=map(tuple,pair)
        if a not in by or b not in by or a==b or a in used or b in used:raise ValueError('Unknown or multiply connected port')
        if by[a].kind!=by[b].kind:raise ValueError('Incompatible port families')
        used.update((a,b))
    if used!=set(by):raise ValueError('Unconnected ports require explicit boundary components')
    return True
