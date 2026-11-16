"""Emit commands/dependencies only. Never launch processes or allocate a GPU."""
import argparse
from pathlib import Path
from ablation_common import HERE, OUT, PYTHON, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['short','long'],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=OUT/a.kind;jobs=[]
    def add(name,script,args,deps,gpu):
        jobs.append(dict(name=name,command=[PYTHON,str(HERE/script),*map(str,args)],dependencies=deps,
                         gpus=gpu,environment=dict(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1'),
                         scheduler_must_set_CUDA_VISIBLE_DEVICES=gpu>0))
    add('train_no_semantic','no_semantic.py',['--kind',a.kind,'--output',root/'no_semantic_head'],[],1)
    for shard in range(4):
        add(f'scores_{shard}','cache_scores.py',['--kind',a.kind,'--shard',shard,'--shards',4,'--output',root/f'scores/shard_{shard}'],[],1)
        add(f'no_semantic_scores_{shard}','cache_no_semantic.py',['--kind',a.kind,'--head',root/'no_semantic_head/best.pt',
            '--shard',shard,'--shards',4,'--output',root/f'no_semantic_scores/shard_{shard}'],['train_no_semantic'],1)
    add('freeze_masks','build_policies.py',['--kind',a.kind,'--score-root',root/'scores','--no-semantic-root',root/'no_semantic_scores',
        '--output',root/'policies'],[f'scores_{s}' for s in range(4)]+[f'no_semantic_scores_{s}' for s in range(4)],0)
    # Main variants use the same codec. The no-semantic variant runs separately
    # with NoSemanticInputs so its inference really does not instantiate B0.
    names=['teacher_dynamic96','rank_generation','rank_legacy','rank_gradient_entropy','rank_uniform',
           'rank_random_seed1','rank_random_seed2','rank_random_seed3']
    names += [f'fixed{k}' for k in [0,32,64,96,128,196,256]]
    names += ['dynamic64','dynamic128']+[f'matched_mean{k}' for k in [64,96,128]]
    names += [f'shuffled{k}_seed{s}' for k in [64,96,128] for s in [1,2,3]]
    names += ['U_zero_recalibrated','same_local_top_p']
    names += [f'without_{term}_fixed96' for term in ['gain','white','cam','density']]
    for label,group in [('main',names),('no_semantic',['no_semantic_retrained_fixed96'])]:
        for shard in range(4):
            add(f'quality_{label}_{shard}','evaluate_masks.py',['--policies',root/'policies','--shard',shard,'--shards',4,
                '--output',root/f'quality_{label}/shard_{shard}','--names',*group],['freeze_masks'],1)
        add(f'merge_{label}','merge_results.py',['--quality-root',root/f'quality_{label}','--output',root/f'{label}_results.json'],
            [f'quality_{label}_{s}' for s in range(4)],0)
    write_json(a.output,dict(status='prepared_not_launched',kind=a.kind,source_directory=str(HERE),output_root=str(root),
        maximum_total_owned_physical_GPUs=4,requires_parent_GPU_grant=True,
        note='existing CFG owns allowed GPUs; this file is declarative and does not start jobs',jobs=jobs,
        expected_counts=dict(main_variants=len(names),no_semantic_variants=1,official_validation_images_per_variant=50000),
        result_files=[str(root/'main_results.json'),str(root/'no_semantic_results.json')]))


if __name__=='__main__':main()
