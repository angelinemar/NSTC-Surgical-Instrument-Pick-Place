"""Render local, shareable plots and tables from the full DP camera evaluation."""
import argparse
import csv
import html
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.export_detection import CAMERAS, CLASSES


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);a=p.parse_args()
    report=json.loads((a.folder/'metrics.json').read_text()); threshold=report['selected_threshold']
    rows=[]
    for split,results in report['splits'].items():
        for camera,result in [('all',results['overall']),*results['per_camera'].items()]:
            rows.append(dict(split=split,camera=camera,category='all',**{k:v for k,v in result.items() if k!='per_class'}))
            for name,metrics in result['per_class'].items():rows.append(dict(split=split,camera=camera,category=name,**metrics))
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (a.folder/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    plt.rcParams.update({'font.size':11})
    fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained')
    for ax,(split,result) in zip(axes,report['splits'].items()):
        x=np.arange(len(CAMERAS));width=.19
        for offset,(key,label) in enumerate([('mAP_50_95','mAP 50:95'),('precision','Precision'),('recall','Recall'),('F1','F1')]):
            vals=[result['per_camera'][c][key] for c in CAMERAS]
            ax.bar(x+(offset-1.5)*width,vals,width,label=label)
        ax.set_xticks(x,[c.replace('cam_','') for c in CAMERAS],rotation=25);ax.set_ylim(0,105)
        ax.set_title(f'{split.title()} — all frames');ax.set_ylabel('Percent');ax.grid(axis='y',alpha=.2)
    axes[1].legend(loc='lower right');fig.suptitle(f'RF-DETR Small: different-camera evaluation | confidence {threshold:.2f}, P/R/F1 IoU ≥ 0.50')
    fig.savefig(a.folder/'camera_metrics.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(14,19),layout='constrained')
    for camera,ax in zip(CAMERAS,axes.flat):
        data=np.load(a.folder/(camera+'_example.npz'));ax.imshow(data['rgb']);ax.axis('off')
        for box in json.loads(str(data['ground_truth'])):
            x,y,w,h=box['bbox'];ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='#35ff7c',lw=1.5))
        for box in json.loads(str(data['predictions'])):
            if box['score']<threshold:continue
            x,y,w,h=box['bbox'];ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='#ff9a35',lw=1.2,linestyle='--'))
            ax.text(x,y,f"{CLASSES[box['category_id']-1]} {box['score']:.2f}",fontsize=6,color='white',
                bbox=dict(facecolor='black',alpha=.6,pad=.8))
        ax.set_title(camera+' | green: ground truth, orange: prediction')
    fig.suptitle('Recorded test-camera examples — same frozen checkpoint',fontsize=17)
    fig.savefig(a.folder/'camera_predictions.png',dpi=130);plt.close(fig)
    lines=['# Evaluation on DP validation/test camera recordings','',
        f"All {sum(report['images'].values()):,} frames; confidence {threshold:.2f} selected using validation only.",
        '','Percentages below are detection metrics, not classification accuracy.','',
        '| Split | Camera | mAP 50:95 | mAP 50 | Precision | Recall | F1 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for split,result in report['splits'].items():
        for camera,r in [('ALL',result['overall']),*result['per_camera'].items()]:
            lines.append('| '+split+' | '+camera+' | '+' | '.join(f'{r[k]:.2f}%' for k in ['mAP_50_95','mAP_50','precision','recall','F1'])+' |')
    lines+=['','Precision = TP / (TP + FP). Recall = TP / (TP + FN).',
        'F1 = 2 × Precision × Recall / (Precision + Recall).',
        'A true positive requires the correct class and IoU ≥ 0.50 with one unmatched ground-truth instance.',
        'mAP 50:95 averages per-class interpolated AP across IoU thresholds 0.50 to 0.95.',
        'There is no meaningful fixed count of background true negatives for object detection.',
        '', 'AP: 101 recall points, maxDets 100/category/image, score floor 0.001, visible objects ≥8 pixels.',
        'The report also includes fixed confidence 0.50 metrics in metrics.json.',
        'These are simulated camera images; adjacent video frames are correlated.',
        'The checkpoint was selected on the original static validation set; this evaluation does not retrain it.',
        '', '![Camera metrics](camera_metrics.png)','', '![Camera predictions](camera_predictions.png)']
    (a.folder/'REPORT.md').write_text('\n'.join(lines)+'\n')
    tables=[]
    for split,result in report['splits'].items():
        body=''.join('<tr><td>'+html.escape(camera)+'</td>'+''.join(f'<td>{r[k]:.2f}%</td>' for k in ['mAP_50_95','mAP_50','precision','recall','F1'])+'</tr>' for camera,r in [('ALL',result['overall']),*result['per_camera'].items()])
        tables.append(f'<h2>{split.title()}</h2><table><tr><th>Camera</th><th>mAP 50:95</th><th>mAP 50</th><th>Precision</th><th>Recall</th><th>F1</th></tr>{body}</table>')
    (a.folder/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>DP camera evaluation</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px;color:#172b40;background:#f5f7fa}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}img{width:100%;margin:20px 0}p{line-height:1.6}</style><h1>RF-DETR Small · DP camera evaluation</h1>'+f'<p>{sum(report["images"].values()):,} images from six cameras. Confidence {threshold:.2f} chosen on validation, frozen for test. Percentages are detection metrics, not classification accuracy.</p>'+''.join(tables)+'<img src="camera_metrics.png"><h2>Actual test recordings</h2><p>Green: ground truth. Orange: model prediction. Detailed per-class results: <a href="metrics.csv">CSV</a> · <a href="metrics.json">JSON</a></p><img src="camera_predictions.png"><p>Precision = TP/(TP+FP). Recall = TP/(TP+FN). F1 = 2PR/(P+R). Matching uses the correct class and IoU ≥ 0.50. mAP 50:95 averages AP across IoU 0.50–0.95 and classes. These simulated video frames are correlated; this does not establish real-camera performance.</p>')
    print(a.folder/'report.html')

if __name__=='__main__':main()
