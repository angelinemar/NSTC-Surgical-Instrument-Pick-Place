"""Validate and train RF-DETR on a P4 standalone detection export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SPLITS = ('train', 'valid', 'test')
CLASSES = ('scalpel', 'scissor', 'love_retractor', 'kelly', 'scalpel_type2')
MODELS = ('nano', 'small', 'medium', 'large')


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_coco(coco: dict[str, Any], split_root: Path, require_size: int = 448) -> dict[str, Any]:
    images = coco.get('images', [])
    annotations = coco.get('annotations', [])
    categories = coco.get('categories', [])
    _require(images, f'{split_root.name}: no images')
    _require(annotations, f'{split_root.name}: no visible object annotations')
    _require([c.get('id') for c in categories] == list(range(1, 6)),
             f'{split_root.name}: category IDs must be 1..5')
    _require(tuple(c.get('name') for c in categories) == CLASSES,
             f'{split_root.name}: unexpected category names/order')

    image_ids: set[int] = set()
    image_by_id: dict[int, dict[str, Any]] = {}
    for image in images:
        image_id = image.get('id')
        _require(isinstance(image_id, int) and image_id not in image_ids,
                 f'{split_root.name}: invalid or duplicate image ID {image_id}')
        image_ids.add(image_id)
        image_by_id[image_id] = image
        width, height = image.get('width'), image.get('height')
        _require((width, height) == (require_size, require_size),
                 f'{split_root.name}: image {image_id} is {width}x{height}; expected native {require_size}x{require_size}')
        relative = Path(str(image.get('file_name', '')))
        _require(not relative.is_absolute() and '..' not in relative.parts,
                 f'{split_root.name}: unsafe image path for ID {image_id}')
        _require((split_root / relative).is_file(),
                 f'{split_root.name}: missing image {relative}')

    annotation_ids: set[int] = set()
    per_class = {name: 0 for name in CLASSES}
    for annotation in annotations:
        annotation_id = annotation.get('id')
        _require(isinstance(annotation_id, int) and annotation_id not in annotation_ids,
                 f'{split_root.name}: invalid or duplicate annotation ID {annotation_id}')
        annotation_ids.add(annotation_id)
        image_id = annotation.get('image_id')
        category_id = annotation.get('category_id')
        _require(image_id in image_by_id, f'{split_root.name}: annotation references missing image {image_id}')
        _require(category_id in range(1, 6), f'{split_root.name}: invalid category {category_id}')
        bbox = annotation.get('bbox')
        _require(isinstance(bbox, list) and len(bbox) == 4,
                 f'{split_root.name}: invalid bbox in annotation {annotation_id}')
        x, y, width, height = bbox
        image = image_by_id[image_id]
        _require(all(isinstance(value, (int, float)) for value in bbox) and
                 x >= 0 and y >= 0 and width > 0 and height > 0 and
                 x + width <= image['width'] and y + height <= image['height'],
                 f'{split_root.name}: out-of-bounds bbox in annotation {annotation_id}')
        _require(annotation.get('iscrowd') == 0,
                 f'{split_root.name}: P4 rigid instances must have iscrowd=0')
        per_class[CLASSES[category_id - 1]] += 1
    _require(all(per_class.values()), f'{split_root.name}: every class needs at least one visible instance; counts={per_class}')
    return {'images': len(images), 'annotations': len(annotations), 'instances_by_class': per_class}


def validate_dataset(dataset: Path, require_size: int = 448) -> dict[str, Any]:
    dataset = dataset.resolve()
    manifest_path = dataset / 'manifest.json'
    _require(manifest_path.is_file(), 'Missing detector manifest.json; pass <EXPORT_DIRECTORY>\\detection')
    manifest = json.loads(manifest_path.read_text())
    _require(manifest.get('export_complete') is True, 'Detection export is incomplete')
    _require(manifest.get('split_rule') in ('independent_run_no_frame_split',
             'committed_episode_or_legacy_session_no_frame_split',
             'static_train_heldout_pick_place'), 'Unsafe split contract')
    _require(manifest.get('rfdetr_dataset_file') == 'roboflow', 'Export predates RF-DETR contract; export again')
    report: dict[str, Any] = {'dataset': str(dataset), 'native_size': require_size, 'splits': {}}
    for split in SPLITS:
        split_root = dataset / split
        annotation_path = split_root / '_annotations.coco.json'
        _require(annotation_path.is_file(), f'{split}: missing _annotations.coco.json')
        report['splits'][split] = validate_coco(json.loads(annotation_path.read_text()), split_root, require_size)
    return report


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, 'item'):
        return value.item()
    return value if isinstance(value, (str, int, float, bool)) or value is None else str(value)


def train(args: argparse.Namespace, audit: dict[str, Any]) -> dict[str, Any]:
    try:
        from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
    except ImportError as exc:
        raise RuntimeError('RF-DETR is not installed. Run: .\\RUNME.ps1 -Mode rfdetr-setup') from exc
    classes = {'nano': RFDETRNano, 'small': RFDETRSmall,
               'medium': RFDETRMedium, 'large': RFDETRLarge}
    output = args.output.resolve()
    _require(not output.exists() or not any(output.iterdir()),
             f'Output must be new or empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    model = classes[args.model]()
    model.train(dataset_dir=str(args.dataset.resolve()), dataset_file='roboflow',
                epochs=args.epochs, batch_size=args.batch_size,
                grad_accum_steps=args.grad_accum_steps, output_dir=str(output))
    checkpoint = output / 'checkpoint_best_total.pth'
    _require(checkpoint.is_file(), f'Training ended without expected checkpoint: {checkpoint}')
    # Score the validation-selected checkpoint, not merely the final in-memory
    # epoch. Using the same variant also preserves its native model resolution.
    best_model = classes[args.model](pretrain_weights=str(checkpoint))
    metrics = best_model.evaluate(dataset_dir=str(args.dataset.resolve()),
                                  dataset_file='roboflow', split='test')
    result = {'contract': 'p4_rfdetr_result_v1', 'training_complete': True,
              'deployment_ready': False, 'model': args.model,
              'dataset_audit': audit, 'checkpoint': str(checkpoint),
              'test_metrics': _jsonable(metrics)}
    (output / 'p4_rfdetr_result.json').write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True,
                        help='The detection directory, not the combined export root')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model', choices=MODELS, default='small')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--grad-accum-steps', type=int, default=4)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.grad_accum_steps < 1:
        parser.error('epochs, batch size, and gradient accumulation must be positive')
    audit = validate_dataset(args.dataset)
    if args.check_only:
        print(json.dumps(audit, indent=2))
        return
    if args.output is None:
        parser.error('--output is required unless --check-only is used')
    print(json.dumps(train(args, audit), indent=2))


if __name__ == '__main__':
    main()
