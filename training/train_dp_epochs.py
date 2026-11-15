"""Epoch-based multi-instrument DP training; checkpoint selection uses validation only."""
import argparse,contextlib,csv,hashlib,json,math,os,random,time,uuid
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from training.sensor_policy import SensorDataset,SensorPolicy,fit_training_stats
from training.perception import save_from_joint

def stopping_state(value,best,bad,min_delta):
    if not math.isfinite(value):raise RuntimeError('Nonfinite validation loss')
    improved=value < best-min_delta
    return (value,0,True) if improved else (best,bad+1,False)

def atomic_save(value,path):
    tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);tmp.replace(path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--skill',choices=['pick','place'],required=True);p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--patience',type=int,default=15);p.add_argument('--min-delta',type=float,default=1e-4)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda');p.add_argument('--resume',type=Path)
    p.add_argument('--wandb',choices=['online','offline','disabled'],default='online')
    p.add_argument('--entity',default='angelinemarcellinachandrawijaya');p.add_argument('--project',default='nstc-surgical-diffusion-policy')
    p.add_argument('--smoke',action='store_true');p.add_argument('--max-train-batches',type=int);p.add_argument('--max-val-batches',type=int)
    a=p.parse_args()
    if min(a.epochs,a.batch_size,a.patience)<1 or a.workers<0:p.error('Invalid positive-count argument')
    if (a.max_train_batches or a.max_val_batches) and not a.smoke:p.error('Batch limits require --smoke; cannot label a partial epoch production training')
    if a.output.exists() and any(a.output.iterdir()) and not a.resume:p.error('Output exists; use --resume or a new output directory')
    a.output.mkdir(parents=True,exist_ok=True);a.dataset=a.dataset.resolve()
    torch.set_num_threads(4);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    cuda=a.device.startswith('cuda')
    if cuda and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if cuda and not torch.cuda.is_bf16_supported():raise RuntimeError('This launcher requires CUDA BF16 support')
    manifest=json.loads((a.dataset/'manifest.json').read_text());mh=hashlib.sha256((a.dataset/'manifest.json').read_bytes()).hexdigest()
    if manifest.get('policy_image_size') not in (224,448):raise ValueError('Unsupported dataset resolution')
    if manifest.get('smoke_only') and not a.smoke:raise ValueError('Smoke export cannot be used for production training')
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()};config.update(manifest_sha256=mh,architecture='SensorPolicy_CNN_ConditionalUnet1D',rgb_size=manifest["policy_image_size"],observation_horizon=2,prediction_horizon=16,precision='bf16-mixed' if cuda else 'fp32',monitor='val/diffusion_loss',classes=manifest.get('task_targets'),model_B_is_input=False)
    resume=torch.load(a.resume,map_location='cpu',weights_only=False) if a.resume else None
    if resume:
        for key in ['manifest_sha256','skill','batch_size','seed','lr','smoke','patience','min_delta']:
            if resume['config'][key]!=config[key]:raise ValueError('Resume configuration mismatch: '+key)
    stats=resume['stats'] if resume else fit_training_stats(a.dataset/(a.skill+'.hdf5'))
    train=SensorDataset(a.dataset,a.skill,'train',stats);valid=SensorDataset(a.dataset,a.skill,'valid',stats,stride=16)
    generator=torch.Generator().manual_seed(a.seed)
    kwargs=dict(num_workers=a.workers,pin_memory=cuda,persistent_workers=a.workers>0)
    if a.workers:kwargs.update(prefetch_factor=1,multiprocessing_context='spawn')
    loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,generator=generator,**kwargs)
    vloader=DataLoader(valid,batch_size=min(a.batch_size,8),shuffle=False,**kwargs)
    model=SensorPolicy().to(a.device);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-6)
    best=float('inf');bad=0;start=0;step=0;best_epoch=0
    if resume:
        model.load_state_dict(resume['model']);opt.load_state_dict(resume['optimizer']);start=resume['epoch'];step=resume['global_step'];best=resume['best'];bad=resume['bad_epochs'];best_epoch=resume['best_epoch']
        torch.set_rng_state(resume['torch_rng']);np.random.set_state(resume['numpy_rng']);random.setstate(resume['python_rng']);generator.set_state(resume['loader_rng'])
        if cuda:torch.cuda.set_rng_state_all(resume['cuda_rng'])
    if start>=a.epochs or bad>=a.patience:raise ValueError('Resume already reached epoch/patience limit; do not silently restart')
    config['train_windows']=len(train);config['valid_windows']=len(valid)
    (a.output/'training_config.json').write_text(json.dumps(config,indent=2))
    import wandb
    wid=resume['wandb_id'] if resume else uuid.uuid4().hex[:8]
    run=wandb.init(project=a.project,entity=a.entity,mode=a.wandb,dir=str(a.output),id=wid,resume='must' if resume and a.wandb=='online' else None,name=('SMOKE_' if a.smoke else '')+f'dp_{a.skill}_ALL5_rgb{config["rgb_size"]}_b{a.batch_size}_seed{a.seed}_{a.output.name}',config=config)
    print(json.dumps(dict(event='ready',skill=a.skill,train_windows=len(train),validation_windows=len(valid),batch_size=a.batch_size,epochs=a.epochs,wandb_url=getattr(run,'url',None))),flush=True)
    def amp():return torch.autocast('cuda',dtype=torch.bfloat16) if cuda else contextlib.nullcontext()
    def move(batch):return {k:v.to(a.device,non_blocking=True) for k,v in batch.items()}
    def checkpoint(epoch):
        return dict(policy_image_size=config['rgb_size'],model=model.state_dict(),optimizer=opt.state_dict(),stats=stats,skill=a.skill,epoch=epoch,global_step=step,best=best,bad_epochs=bad,best_epoch=best_epoch,config=config,wandb_id=wid,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if cuda else [],numpy_rng=np.random.get_state(),python_rng=random.getstate(),loader_rng=generator.get_state(),observation_horizon=2,prediction_horizon=16,control_dt_s=.02,input_contract='p4_sensor_task_v2',smoke_only=a.smoke,controller_contract=manifest['controller_contract'])
    history=a.output/'metrics.csv';fields=['epoch','train/loss','train/diffusion_loss','train/perception_loss','val/loss','val/diffusion_loss','val/perception_loss','best_epoch','bad_epochs','seconds']
    try:
        for epoch in range(start+1,a.epochs+1):
            began=time.monotonic();model.train();sums=np.zeros(3);seen=0
            for i,batch in enumerate(loader):
                batch=move(batch);opt.zero_grad(set_to_none=True)
                with amp():loss,dp,seg=model.losses(batch)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
                opt.step();step+=1;b=len(batch['rgb']);values=np.array([loss.item(),dp.item(),seg.item()]);sums+=values*b;seen+=b
                if i%20==0:
                    print(f'{a.skill} epoch {epoch}/{a.epochs} batch {i+1}/{len(loader)} loss={values[0]:.5f} diffusion={values[1]:.5f}',flush=True);run.log({'train/step_loss':values[0],'train/step_diffusion_loss':values[1],'epoch':epoch},step=step)
                if a.max_train_batches and i+1>=a.max_train_batches:break
            model.eval();vs=np.zeros(3);vn=0
            # Fixed noise and timesteps make epoch comparisons reproducible, without changing training RNG.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()] if cuda else []),torch.no_grad():
                torch.manual_seed(a.seed+10000)
                for i,batch in enumerate(vloader):
                    batch=move(batch)
                    with amp():vals=model.losses(batch)
                    b=len(batch['rgb']);vs+=np.array([v.item() for v in vals])*b;vn+=b
                    if i%50==0:print(f'{a.skill} validation {i+1}/{len(vloader)}',flush=True)
                    if a.max_val_batches and i+1>=a.max_val_batches:break
            tr=sums/seen;va=vs/vn;best,bad,improved=stopping_state(float(va[1]),best,bad,a.min_delta)
            if improved:best_epoch=epoch
            state=checkpoint(epoch)
            if improved:
                atomic_save(state,a.output/'best.ckpt')
                portable={k:state[k] for k in ['policy_image_size','model','stats','skill','observation_horizon','prediction_horizon','control_dt_s','input_contract','smoke_only','controller_contract']}
                atomic_save(portable,a.output/'checkpoint_best.pt')
                save_from_joint(model,a.output/'perception_best.pt',a.smoke)
            atomic_save(state,a.output/'last.ckpt')
            row=dict(zip(fields,[epoch,*tr,*va,best_epoch,bad,time.monotonic()-began]));exists=history.exists()
            with history.open('a',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader() if not exists else None;w.writerow(row)
            run.log(row,step=step);print(json.dumps(row),flush=True)
            if bad>=a.patience:print('Early stopping: no validation diffusion-loss improvement.',flush=True);break
        result=dict(peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30 if cuda else None,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30 if cuda else None,training_complete=True,smoke_only=a.smoke,completed_epochs=epoch,best_epoch=best_epoch,best_validation_diffusion_loss=best,stop_reason='early_stopping' if bad>=a.patience else 'max_epochs',checkpoint=str(a.output/'checkpoint_best.pt'),resume_checkpoint=str(a.output/'last.ckpt'),wandb_url=getattr(run,'url',None),test_used_for_selection=False,physical_rollout_verified=False,ready_for_deployment=False)
        (a.output/'result.json').write_text(json.dumps(result,indent=2));run.summary.update(result);print(json.dumps(result),flush=True)
    finally:run.finish()
if __name__=='__main__':main()
