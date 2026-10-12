"""Reproducible RF-DETR-Seg Small training on the frozen Trial 02 dataset."""
import argparse,json,hashlib
from pathlib import Path
from training.rfdetr_pipeline import _jsonable


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--grad-accum-steps',type=int,default=16);p.add_argument('--resolution',type=int,default=768)
    p.add_argument('--num-workers',type=int,default=4);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--wandb',action='store_true');p.add_argument('--run-name',default='trial02-seg-small-768')
    p.add_argument('--resume',type=Path);p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    if a.resolution%24 or min(a.epochs,a.batch_size,a.grad_accum_steps,a.resolution)<=0:p.error('Positive settings and resolution divisible by 24 required')
    manifest_path=a.dataset/'manifest.json';m=json.loads(manifest_path.read_text())
    assert m['contract']=='nstc_seg_dataset_v1' and m['export_complete']
    assert m['split_rule'] in ('configuration_disjoint_hemisphere_lighting_v1','synthetic_train_original_dp_valid_test_v1')
    for split in ('train','valid','test'):
        coco=json.loads((a.dataset/split/'_annotations.coco.json').read_text())
        assert coco['images'] and coco['annotations']
        for annotation in coco['annotations']:assert annotation.get('segmentation')
        for image in coco['images']:assert (a.dataset/split/image['file_name']).is_file()
    if a.check_only:print('Segmentation dataset contract and files verified');return
    if a.output.exists() and any(a.output.iterdir()) and not a.resume:raise ValueError('Use a new output directory or explicit --resume last.ckpt')
    if a.resume and (not a.resume.is_file() or a.resume.suffix!='.ckpt'):raise ValueError('Resume requires full Lightning .ckpt')
    a.output.mkdir(parents=True,exist_ok=True)
    config={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config['dataset_manifest_sha256']=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    config_path=a.output/'trial_config.json'
    if a.resume and config_path.exists():
        old=json.loads(config_path.read_text())
        for key in ('dataset_manifest_sha256','resolution','seed'):
            if old[key]!=config[key]:raise ValueError('Resume configuration differs: '+key)
    config_path.write_text(json.dumps(config,indent=2))
    from rfdetr import RFDETRSegSmall
    model=RFDETRSegSmall(resolution=a.resolution)
    model.train(dataset_dir=str(a.dataset.resolve()),dataset_file='roboflow',output_dir=str(a.output.resolve()),
        epochs=a.epochs,batch_size=a.batch_size,grad_accum_steps=a.grad_accum_steps,num_workers=a.num_workers,
        resolution=a.resolution,multi_scale=False,expanded_scales=False,seed=a.seed,lr=1e-4,
        use_ema=True,early_stopping=True,early_stopping_patience=15,checkpoint_interval=5,
        tensorboard=True,wandb=a.wandb,project='nstc-surgical-segmentation',run=a.run_name,
        log_per_class_metrics=True,run_test=False,resume=str(a.resume.resolve()) if a.resume else None)
    checkpoint=a.output/'checkpoint_best_total.pth'
    assert checkpoint.is_file(),'Missing best checkpoint'
    best=RFDETRSegSmall.from_checkpoint(str(checkpoint))
    metrics=best.evaluate(dataset_dir=str(a.dataset.resolve()),dataset_file='roboflow',split='test',
        output_dir=str(a.output/'heldout_test'),resolution=a.resolution,batch_size=a.batch_size,
        num_workers=a.num_workers,log_per_class_metrics=True)
    (a.output/'result.json').write_text(json.dumps(dict(training_complete=True,checkpoint=str(checkpoint.resolve()),
        task='instance_segmentation',test_metrics=_jsonable(metrics),production_ready=False),indent=2))
    print(json.dumps(_jsonable(metrics),indent=2))

if __name__=='__main__':main()
