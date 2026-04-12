from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np

@dataclass
class TaskResult:
    score: float
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)

class Task1_BasicPlacement:
    TASK_ID = 'task1_basic_placement'
    DIFFICULTY = 'easy'
    MIN_FRAMES = 10
    REWARD_THRESH = 0.5
    SUCCESS_RATIO = 0.70

    def __init__(self):
        self.rewards = []

    def reset(self):
        self.rewards = []

    def update(self, reward, info):
        self.rewards.append(float(reward))

    def grade(self):
        if len(self.rewards) < self.MIN_FRAMES:
            return TaskResult(0., False, {'error': f'Need >={self.MIN_FRAMES} frames, got {len(self.rewards)}'})
        passed_frames = sum(1 for r in self.rewards if r >= self.REWARD_THRESH)
        ratio = float(passed_frames) / len(self.rewards)
        return TaskResult(round(ratio, 4), ratio >= self.SUCCESS_RATIO, {'ratio': ratio})

class Task2_RealisticBlend:
    TASK_ID = 'task2_realistic_blend'
    DIFFICULTY = 'medium'
    MIN_FRAMES = 15
    ALIGN_TH = 0.60
    LIGHT_TH = 0.60
    SUCCESS_RATIO = 0.60

    def __init__(self):
        self.frames = []

    def reset(self):
        self.frames = []

    def update(self, reward, info):
        rc = info.get('reward_components', {})
        self.frames.append({
            'a': rc.get('alignment', 0.),
            'l': rc.get('lighting', 0.)
        })

    def grade(self):
        if len(self.frames) < self.MIN_FRAMES:
            return TaskResult(0., False, {'error': f'Need >={self.MIN_FRAMES} frames, got {len(self.frames)}'})
        passed = sum(1 for f in self.frames if f['a'] >= self.ALIGN_TH and f['l'] >= self.LIGHT_TH)
        ratio = float(passed) / len(self.frames)
        return TaskResult(round(ratio, 4), ratio >= self.SUCCESS_RATIO, {'ratio': ratio})

class Task3_TemporalConsistency:
    TASK_ID = 'task3_temporal_consistency'
    DIFFICULTY = 'hard'
    MIN_FRAMES = 30
    MEAN_TH = 0.75
    MIN_FRAME = 0.30
    OCC_TH = 0.70
    OCC_RATIO = 0.80
    STD_TH = 0.15

    def __init__(self):
        self.rewards = []
        self.temporals = []

    def reset(self):
        self.rewards = []
        self.temporals = []

    def update(self, reward, info):
        self.rewards.append(float(reward))
        self.temporals.append(float(info.get('reward_components', {}).get('temporal', 0.)))

    def grade(self):
        if len(self.rewards) < self.MIN_FRAMES:
            return TaskResult(0., False, {'error': f'Need >={self.MIN_FRAMES} frames, got {len(self.rewards)}'})
        arr = np.array(self.rewards)
        mean_r = float(arr.mean())
        std_r = float(arr.std())
        min_r = float(arr.min())
        occ_pass = sum(t >= self.OCC_TH for t in self.temporals) / len(self.temporals)
        checks = [
            mean_r >= self.MEAN_TH,
            min_r >= self.MIN_FRAME,
            occ_pass >= self.OCC_RATIO,
            std_r <= self.STD_TH
        ]
        score = (float(np.clip(mean_r/self.MEAN_TH, 0, 1)) +
                 float(np.clip(max(0, min_r)/self.MIN_FRAME, 0, 1)) +
                 float(np.clip(occ_pass/self.OCC_RATIO, 0, 1)) +
                 float(np.clip(max(0, 1.-std_r/self.STD_TH), 0, 1))) / 4
        return TaskResult(round(float(score), 4), all(checks), {
            'mean': mean_r, 'min': min_r, 'occ_pass': occ_pass, 'std': std_r
        })

TASKS = {
    Task1_BasicPlacement.TASK_ID: Task1_BasicPlacement,
    Task2_RealisticBlend.TASK_ID: Task2_RealisticBlend,
    Task3_TemporalConsistency.TASK_ID: Task3_TemporalConsistency
}
