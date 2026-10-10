"""Bounded settling with sustained velocity checks; never freeze moving tools."""
import math


def settle(step, measure, minimum_steps=120, maximum_steps=1200, stable_steps=30,
           interval=10, speed_limit=.015):
    stable=0
    speeds={}
    for steps in range(1,maximum_steps+1):
        step()
        if steps<minimum_steps or steps%interval:
            continue
        speeds=measure()
        if not speeds or any(not math.isfinite(speed) for speed in speeds.values()):
            raise RuntimeError('Invalid instrument velocity while settling')
        stable=stable+interval if max(speeds.values())<=speed_limit else 0
        if stable>=stable_steps:
            return dict(steps=steps,stable_steps=stable,speed_limit=speed_limit,speeds=speeds)
    worst=max(speeds,key=speeds.get)
    raise RuntimeError(f'Unsettled instrument: {worst}; speed={speeds[worst]:.6f} m/s after {maximum_steps} physics steps (limit {speed_limit} m/s)')
