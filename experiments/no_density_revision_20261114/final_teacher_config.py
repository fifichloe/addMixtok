"""Resolve the final retained score terms only after the component design lock."""
import json
from pathlib import Path
from density_free_config import HERE, ROOT, DATA, VAL_PATHS, OLD, FRESH, HEADS, KINDS, CAM, UTILITY_WEIGHT
from density_free_config import OUTPUT as REVISION_OUTPUT, sha, write_json, source_identities as original_identities

OUTPUT = REVISION_OUTPUT/'final'


def load_design():
    path=HERE/'design_lock.json'
    value=json.loads(path.read_text())
    assert value['status']=='density_free_component_design_locked'
    assert value['density_component_removed'] is True and set(value['teachers'])==set(KINDS)
    for kind in KINDS:
        record=value['teachers'][kind]
        assert record['white_direct_weight'] in (0.,.5)
        assert record['gain_weight']==1. and record['cam_weight']==CAM[kind]
        assert record['utility_weight']==UTILITY_WEIGHT[kind]
        expected='without_white_dynamic96' if record['white_direct_weight']==0. else 'full_dynamic96'
        assert record['selected_policy_name']==expected
        policy_path=HERE/'design_gate'/kind/'policies/protocol.json'
        assert record['teacher_policy_sha256']==sha(policy_path)
        protocol=json.loads(policy_path.read_text())
        policy=next(row for row in protocol['variants'] if row['name']==expected)
        assert policy['calibration']['threshold']==record['threshold']
        audit_path=HERE/'design_gate'/kind/'quality_independent_audit.json'
        audit=json.loads(audit_path.read_text())
        assert audit['status']=='all9_full50000_component_qualities_independently_verified'
        assert record['component_quality_audit_sha256']==sha(audit_path)
        assert audit['protocol_sha256']==sha(policy_path)
    return value


def source_identities():
    result=original_identities()
    result.update({str(path):sha(path) for path in [HERE/'final_teacher_config.py',
                  HERE/'teacher_final.py',HERE/'design_lock.json']})
    return result


def policies():
    design=load_design();result={}
    for kind in KINDS:
        record=design['teachers'][kind]
        audit=json.loads((HERE/'design_gate'/kind/'quality_independent_audit.json').read_text())
        result[kind]=dict(name=f"{kind}_density_free_white{record['white_direct_weight']}",
            weight=UTILITY_WEIGHT[kind],threshold=record['threshold'],density=0.,
            white=record['white_direct_weight'],cam=CAM[kind],
            teacher_rFID=audit['rows'][record['selected_policy_name']]['rFID'],
            component_design_sha256=sha(HERE/'design_lock.json'),
            teacher_policy_sha256=record['teacher_policy_sha256'],
            selected_policy_name=record['selected_policy_name'],
            source='final design frozen after full component controls; original score-only thresholds retained',
            quality_used_for_thresholds=False,
            component_choice_informed_by_ablation_quality=True)
    return result
