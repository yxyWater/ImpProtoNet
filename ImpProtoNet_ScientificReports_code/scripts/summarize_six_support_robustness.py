import csv,gzip,hashlib,json

from collections import Counter,defaultdict

from pathlib import Path

import numpy as np

NAMES={'M0':'ProtoNet','M1':'FrozenNet','M2':'TransferNet','M3':'MatchNet','M5':'RelationNet','Ours':'Ours'}

ESTIMANDS=['AvgAcc','IqrAcc','QCR','QCT_gt1']

def loadgz(path):
    with gzip.open(path,'rt',encoding='utf-8-sig') as f:return [json.loads(line) for line in f]

def query_statistics(counts,truth):
    total=sum(counts.values())
    if total!=100:raise ValueError('Expected exactly 100 support resamples per query')
    return counts.get(truth,0)/total,1-sum(n*(n-1) for n in counts.values())/(total*(total-1)),float(len(counts)>1)

def bootstrap(grid,draws):
    if grid.shape!=(10,4):raise ValueError('Expected ten combination clusters and four estimands')
    replicates=grid[draws].mean(axis=1)
    return grid.mean(0),np.quantile(replicates,[.025,.975],axis=0,method='linear'),replicates

def summarize_run(rows,episodes,shot):
    expected={e['episode_id']:e for e in episodes if e['support_resample_index']>0}
    if len(expected)!=1000:raise ValueError('Expected 1000 distinct resampling episodes')
    seen=set();byquery=defaultdict(Counter);query_identity={};byepisode=defaultdict(list)
    for row in rows:
        if row.get('repeat',-1)==0:continue
        eid=row['episode_id'];e=expected[eid];position=int(row['query_position'])
        key=(eid,position)
        if key in seen:raise ValueError('Duplicate episode/query position')
        seen.add(key)
        assert row['split_id']=='split_01' and int(row['shot'])==shot
        combo=e['disease_combination_index'];rep=e['support_resample_index']
        assert int(row['combination'])==combo and int(row['repeat'])==rep
        queries=[(c['class_id'],x) for c in e['classes'] for x in c['query']]
        truth,item=queries[position-1]
        assert row['patient_id']==item['patient_id'] and row['image_id']==item['image_id']
        assert row['true_class_id']==truth and row['predicted_class_id'] in e['class_ids']
        qkey=combo,position
        identity=(truth,item['patient_id'],item['image_id'])
        assert query_identity.setdefault(qkey,identity)==identity
        byquery[qkey][row['predicted_class_id']]+=1
        byepisode[combo,rep].append(float(row['predicted_class_id']==truth))
    assert len(seen)==15000 and len(byquery)==150 and len(byepisode)==1000
    assert all(len(v)==15 for v in byepisode.values())
    assert all(sum(counts.values())==100 for counts in byquery.values())
    grid=[];query_rows=[];episode_rows=[]
    for combo in range(1,11):
        accuracies=np.array([np.mean(byepisode[combo,rep]) for rep in range(1,101)])
        qs=[query_statistics(byquery[combo,j],query_identity[combo,j][0]) for j in range(1,16)]
        avg=float(accuracies.mean());iqr=float(np.quantile(accuracies,.75)-np.quantile(accuracies,.25))
        grid.append([avg,iqr,float(np.mean([v[1] for v in qs])),float(np.mean([v[2] for v in qs]))])
        assert abs(avg-np.mean([v[0] for v in qs]))<1e-12
        for j,stats in enumerate(qs,1):
            truth,patient,image=query_identity[combo,j]
            query_rows.append(dict(combination=combo,query_position=j,true_class_id=truth,patient_id=patient,image_id=image,
                counts=json.dumps(byquery[combo,j],sort_keys=True),accuracy=stats[0],QCR=stats[1],QCT_gt1=stats[2]))
        episode_rows.extend(dict(combination=combo,repeat=i+1,accuracy=float(v)) for i,v in enumerate(accuracies))
    return np.array(grid),query_rows,episode_rows

def writecsv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
