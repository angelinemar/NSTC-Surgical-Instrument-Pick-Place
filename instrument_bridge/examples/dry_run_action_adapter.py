"""Example adapter entrypoint. Validates the handoff and deliberately moves no robot."""
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python dry_run_action_adapter.py target_request.json')
    request = json.loads(Path(sys.argv[1]).read_text())
    if request.get('contract') != 'nstc_target_request_v1':
        raise ValueError('Unknown target request contract')
    if not Path(request['image_path']).is_file():
        raise FileNotFoundError(request['image_path'])
    result = dict(status='dry_run_only', instruction=request['instruction'],
                  model=request['perception_model'], camera=request['camera'],
                  target_class=request['target']['class_name'],
                  target_box=request['target']['bbox_xyxy'],
                  target_center_px=request['target']['center_xy'],
                  robot_action_executed=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
