"""Full50k current-design token and position-inclusive fixed-width bitrate data."""
import csv
import json
from pathlib import Path
import sys

import numpy as np

REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV))
from final_teacher_config import HERE, OUTPUT, sha, write_json

OUT = HERE/'efficiency/budget_statistics'
DATA = OUTPUT.parent


def summarize(hist):
    hist = np.asarray(hist, dtype=np.int64)
    assert hist.shape == (257,) and np.all(hist >= 0) and hist.sum() == 50000
    positions = np.arange(257)
    mean = float(hist@positions/50000)
    occupied = np.flatnonzero(hist)
    return dict(n=50000, mean_K=mean, mean_content_tokens=32+mean,
        min_K=int(occupied[0]), max_K=int(occupied[-1]),
        K_population_std=float(np.sqrt(hist@np.square(positions-mean)/50000)),
        global_content_bits=384, position_mask_bits=256,
        mean_fixed_width_bits=384+14*mean+256,
        mean_fixed_width_bpp=(384+14*mean+256)/65536,
        alternative_explicit_8bit_indices_bpp=(384+22*mean)/65536,
        histogram=hist.tolist())


def main():
    assert not OUT.exists(), 'No overwrite of current budget statistics'
    design_sha = sha(HERE/'design_lock.json')
    records = {}

    def masks(name, row, provenance):
        assert sha(row['mask_file']) == row['mask_sha256']
        assert sha(row['K_file']) == row['K_sha256']
        k = np.load(row['K_file'])
        mask = np.load(row['mask_file'], mmap_mode='r')
        assert k.shape == (50000,) and mask.shape == (50000, 256)
        assert np.array_equal(mask.sum(1), k)
        assert np.all((k >= 0) & (k <= 256))
        value = summarize(np.bincount(k.astype(np.int64), minlength=257))
        assert value['mean_K'] == row['mean_K']
        records[name] = dict(**value, provenance=provenance,
            policy_mask_sha256=row['mask_sha256'], policy_K_sha256=row['K_sha256'])

    for kind in ('long',):
        path = HERE/f'design_gate/{kind}/policies/protocol.json'
        protocol = json.loads(path.read_text())
        row = next(row for row in protocol['variants'] if row['name'] == 'full_dynamic96')
        masks(f'reconstruction_{kind}_dynamic96', row, dict(protocol=str(path), sha256=sha(path)))
    for kind in ('long',):
        path = HERE/f'reconstruction_controls/{kind}/quality_independent_audit.json'
        audit = json.loads(path.read_text())
        assert audit['status'] == 'all19_current_reconstruction_qualities_independently_verified'
        for name, row in audit['rows'].items():
            masks(f'reconstruction_{kind}_{name}', row, dict(quality_audit=str(path), sha256=sha(path)))

    def generation(label, entry):
        path = Path(entry['summary'])
        assert sha(path) == entry['summary_sha256']
        result = json.loads(path.read_text())
        assert result['status'] == 'complete' and result['n'] == 50000
        assert result['generation_rounds'] == '8+8'
        assert result['router']['component_design_sha256'] == design_sha
        assert result['all_masks_and_K_replayed'] and result['all_routes_and_inputs_saved']
        hist = np.zeros(257, dtype=np.int64)
        inputs = {}
        for rank in range(4):
            folder = path.parent
            receipt_path = folder/f'shard{rank}.json'
            receipt = json.loads(receipt_path.read_text())
            assert receipt['status'] == 'complete'
            for name in ('route_K', 'route_mask'):
                source = folder/f'{name}_shard{rank}.npy'
                assert sha(source) == receipt['dump_hashes'][name]
                inputs[str(source)] = receipt['dump_hashes'][name]
            counts = np.load(folder/f'route_K_shard{rank}.npy').astype(np.int64)
            mask = np.load(folder/f'route_mask_shard{rank}.npy', mmap_mode='r')
            assert np.array_equal(mask.sum(1), counts)
            hist += np.bincount(counts, minlength=257)
            inputs[str(receipt_path)] = sha(receipt_path)
        assert hist.tolist() == result['tokens']['histogram']
        value = summarize(hist)
        assert value['mean_K'] == result['tokens']['mean_generated_2d']
        records[label] = dict(**value, seed=result['seed'], kind=result['kind'],
            checkpoint_sha256=result['checkpoint_sha256'],
            provenance=dict(summary=str(path), summary_sha256=sha(path), actual_inputs_sha256=inputs))

    path = DATA/'generation_evaluation/main_four_sampling_seeds.json'
    main = json.loads(path.read_text())
    assert main['status'] == 'all_four_main_sampling_seeds_complete'
    assert main['component_design_sha256'] == design_sha
    for entry in main['rows']:
        generation(f"generation_main_seed{entry['seed']}", entry)
    path = DATA/'generation_controls/inference/results.json'
    inference = json.loads(path.read_text())
    assert inference['status'] == 'current_router_strictfixed_and_U0_controls_complete'
    for mode, group in inference['modes'].items():
        for entry in group['rows']:
            generation(f"generation_inference_{mode}_seed{entry['seed']}", entry)
    path = DATA/'fixed96_evaluation/results.json'
    fixed = json.loads(path.read_text())
    assert fixed['status'] == 'fixed96_training_four_sampling_seeds_and_common_CFG_baselines_complete'
    for entry in fixed['rows']:
        generation(f"generation_independently_trained_fixed96_seed{entry['seed']}", entry)
    audit_path = HERE/'long_only_execution_20261115/existing_generator_retraining_audit.json'
    old = json.loads(audit_path.read_text())
    assert old['status'] == 'existing_random_uniform_Dense_all20k_and50k_complete_verified'
    for mode, entry in old['checkpoints'].items():
        path = Path(entry['full50k_summary'])
        summary = json.loads(path.read_text())
        assert summary['status'] == 'complete' and summary['n'] == 50000
        assert summary['checkpoint_sha256'] == entry['sha256']
        counts = np.asarray(summary['tokens']['k_histogram'], dtype=np.int64)
        assert counts.sum() == 50000 and counts[entry['fixed_K']] == 50000
        value = summarize(counts)
        if mode == 'dense':
            # Full-grid spatial support is implicit, and global tokens/context
            # do not affect dense output. Keep hybrid endpoint accounting separate.
            value.update(mean_content_tokens=256, global_content_bits=0, position_mask_bits=0,
                mean_fixed_width_bits=14*256, mean_fixed_width_bpp=14*256/65536,
                alternative_explicit_8bit_indices_bpp=None)
        records['generation_reused_'+mode] = dict(**value, seed=summary['seed'],
            checkpoint_sha256=entry['sha256'],
            provenance=dict(summary=str(path), summary_sha256=sha(path), baseline_audit_sha256=sha(audit_path),
                proof='completed full50k fixed-budget histogram and audited checkpoint; not new-router raw mask dumps'),
            dense_native_representation_excludes_unused_stage1=mode == 'dense')
    OUT.mkdir()
    write_json(OUT/'statistics.json', dict(status='current_full50000_budgets_and_bitrate_complete',
        records=records, component_design_sha256=design_sha, codebook_sizes=dict(global_1d=4096, local_2d=16384),
        formula='(32*12 +14*mean(K) +256)/(256*256)',
        accounting='Fixed-width content codewords plus raw256-bit bitmap, whose population count gives K',
        entropy_coding=False, container_header_or_model_weights_included=False,
        bitmap_size_kept_for_all_endpoints=True, actual_compressed_file_size_claimed=False,
        sampling_seeds_not_training_repeats=True, source_sha256=sha(Path(__file__))))
    fields = ['name', 'n', 'mean_K', 'mean_content_tokens', 'min_K', 'max_K', 'mean_fixed_width_bits', 'mean_fixed_width_bpp']
    with (OUT/'summary.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, row in records.items():
            writer.writerow(dict(name=name, **{key:row[key] for key in fields if key != 'name'}))
    for name, row in records.items():
        with (OUT/(name+'.histogram.csv')).open('w') as handle:
            writer = csv.writer(handle)
            writer.writerow(['K', 'images', 'fraction'])
            for k, count in enumerate(row['histogram']):
                writer.writerow([k, count, count/50000])
    print(json.dumps(dict(status='current_full50000_budgets_and_bitrate_complete', records=len(records))), flush=True)


if __name__ == '__main__':
    main()
