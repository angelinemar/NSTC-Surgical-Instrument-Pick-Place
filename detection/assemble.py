"""Assemble static training images with held-out pick/place validation and test."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detection.dataset import write_json
from training.rfdetr_pipeline import validate_coco


def assemble(capture, evaluation, output):
    capture, evaluation, output = map(lambda p: Path(p).resolve(), (capture, evaluation, output))
    if output.exists():
        raise ValueError('Output must be a new folder')
    config = json.loads((capture/'config.json').read_text())
    status = json.loads((capture/'status.json').read_text())
    if config.get('contract') != 'p4_static_detection_v1' or status.get('state') != 'complete':
        raise ValueError('Finish the static collection before assembling training data')
    evaluation_manifest = json.loads((evaluation/'manifest.json').read_text())
    if evaluation_manifest.get('export_complete') is not True or evaluation_manifest.get('split_rule') not in (
        'independent_run_no_frame_split', 'committed_episode_or_legacy_session_no_frame_split'):
        raise ValueError('Evaluation source must be a completed episode-grouped pick/place export')
    sources = {'train': capture/'train', 'valid': evaluation/'valid', 'test': evaluation/'test'}
    records, seen, audit = {}, {}, {}
    for split, source in sources.items():
        coco = json.loads((source/'_annotations.coco.json').read_text())
        audit[split] = validate_coco(coco, source, 448)
        for item in coco['images']:
            relative = Path(item['file_name'])
            path = (source/relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(source):
                raise ValueError('Unsafe image path')
            from PIL import Image
            with Image.open(path) as image:
                image.load()
                if image.size != (448,448) or image.mode != 'RGB':
                    raise ValueError('Expected RGB 448 image: '+str(path))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen and seen[digest] != split:
                raise ValueError('Identical RGB image found across splits: '+split)
            seen[digest] = split
        records[split] = coco
    output.mkdir(parents=True)
    for split, coco in records.items():
        destination = output/split
        destination.mkdir()
        for item in coco['images']:
            target = destination/item['file_name']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sources[split]/item['file_name'], target)
        write_json(destination/'_annotations.coco.json', coco)
    write_json(output/'manifest.json', dict(export_complete=True,
        split_rule='static_train_heldout_pick_place', rfdetr_dataset_file='roboflow',
        production_ready=False, smoke_only=evaluation_manifest.get('smoke_only',False),
        sources={k:str(v) for k,v in sources.items()}, audit=audit,
        note='Source validation/test must already be episode-grouped. Image hash checks do not prove episode independence.'))
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.capture, args.evaluation, args.output), indent=2))
