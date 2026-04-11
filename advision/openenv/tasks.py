from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np

@dataclass
class TaskResult:
    score: float; passed: bool
    details: Dict[str,Any]=field(default_factory=dict)

class Task1_BasicPlacement:
    TASK_ID='task1_basic_placement'; DIFFICULTY='easy'
    MIN_FRAMES=10; REWARD_THRESH=0.5; SUCCESS_RATIO=0.70
    def __init__(self): self.rewards=[]
    def reset(self): self.rewards=[]
    def update(self,reward,info): self.rewards.append(float(reward))
    def grade(self):
        if len(self.rewards)<self.MIN_FRAMES:
            return TaskResult(0.,False,{'error':f'Need >={self.MIN_FRAMES} frames, got {len(self.rewards)}'})
        passing=sum(r>=self.REWARD_THRESH for r in self.rewards)
        ratio=passing/len(self.rewards)
        return TaskResult(round(float(np.clip(ratio/self.SUCCESS_RATIO,0,1)),4),ratio>=self.SUCCESS_RATIO,
                          {'passing_ratio':round(ratio,4),'mean_reward':round(float(np.mean(self.rewards)),4),
                           'passing_frames':passing,'total_frames':len(self.rewards)})

class Task2_RealisticBlend:
    TASK_ID='task2_realistic_blend'; DIFFICULTY='medium'
    MIN_FRAMES=15; ALIGN_TH=0.60; LIGHT_TH=0.60; SUCCESS_RATIO=0.60
    def __init__(self): self.frames=[]
    def reset(self): self.frames=[]
    def update(self,reward,info):
        rc=info.get('reward_components',{})
        self.frames.append({'alignment':float(rc.get('alignment',0.)),'lighting':float(rc.get('lighting',0.)),'reward':float(reward)})
    def grade(self):
        if len(self.frames)<self.MIN_FRAMES:
            return TaskResult(0.,False,{'error':f'Need >={self.MIN_FRAMES} frames'})
        co=sum(1 for f in self.frames if f['alignment']>=self.ALIGN_TH and f['lighting']>=self.LIGHT_TH)
        ratio=co/len(self.frames)
        am=float(np.mean([f['alignment'] for f in self.frames]))
        lm=float(np.mean([f['lighting']  for f in self.frames]))
        score=float(np.clip(0.5*ratio/self.SUCCESS_RATIO+0.5*(am/self.ALIGN_TH+lm/self.LIGHT_TH)/2,0,1))
        return TaskResult(round(score,4),ratio>=self.SUCCESS_RATIO,
                          {'co_passing_ratio':round(ratio,4),'mean_alignment':round(am,4),
                           'mean_lighting':round(lm,4),'co_passing':co,'total':len(self.frames)})

class Task3_TemporalConsistency:
    TASK_ID='task3_temporal_consistency'; DIFFICULTY='hard'
    MIN_FRAMES=30; MEAN_TH=0.75; MIN_FRAME=0.30; OCC_TH=0.70; OCC_RATIO=0.80; STD_TH=0.15
    def __init__(self): self.rewards=[]; self.temporals=[]
    def reset(self): self.rewards=[]; self.temporals=[]
    def update(self,reward,info):
        self.rewards.append(float(reward))
        self.temporals.append(float(info.get('reward_components',{}).get('temporal',0.)))
    def grade(self):
        if len(self.rewards)<self.MIN_FRAMES:
            return TaskResult(0.,False,{'error':f'Need >={self.MIN_FRAMES} frames, got {len(self.rewards)}'})
        arr=np.array(self.rewards)
        mean_r=float(arr.mean()); std_r=float(arr.std()); min_r=float(arr.min())
        occ_pass=sum(t>=self.OCC_TH for t in self.temporals)/len(self.temporals)
        checks=[mean_r>=self.MEAN_TH,min_r>=self.MIN_FRAME,occ_pass>=self.OCC_RATIO,std_r<=self.STD_TH]
        score=(float(np.clip(mean_r/self.MEAN_TH,0,1))+float(np.clip(max(0,min_r)/self.MIN_FRAME,0,1))+
               float(np.clip(occ_pass/self.OCC_RATIO,0,1))+float(np.clip(max(0,1.-std_r/self.STD_TH),0,1)))/4
        return TaskResult(round(float(score),4),all(checks),
                          {'mean_reward':round(mean_r,4),'std_reward':round(std_r,4),
                           'min_reward':round(min_r,4),'occ_passing_ratio':round(occ_pass,4),
                           'checks_passed':int(sum(checks)),'checks_total':len(checks)})

TASKS={'task1':Task1_BasicPlacement,'task2':Task2_RealisticBlend,'task3':Task3_TemporalConsistency}
print('[OK] tasks.py imported')
