#!/usr/bin/env python3
"""Rebuild the 12 active numeric table sources from frozen evidence, without data/models.

Templates contain captions and layout only. Every numeric data cell is supplied
by the frozen tables/counts below. Bootstrap intervals are read, never rerun.
"""
from pathlib import Path
import argparse
import csv
import json
import re
import tempfile

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('plain', 'point', 'plain_desc', 'point_desc')
CONDITIONS = ('source', 'grey', 'mismatch', 'mask', 'noise', 'mirror')

def read(relative):
    with (ROOT / relative).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))

def rq1(name): return read('reports/frozen/rq1_final/data/' + name + '.tsv')
def rq2(name): return read('reports/frozen/rq2_final/tables/' + name + '.tsv')
def fmt(value, digits=4, signed=False):
    text = format(float(value), ('+' if signed else '') + f'.{digits}f')
    if float(text) == 0:
        text = text.removeprefix('-')
    return text.replace('-', '$-$')
def integer(value):return format(int(value), ',').replace('-', '$-$')
def interval(row, lo, hi, digits=4):return '[' + fmt(row[lo], digits) + ', ' + fmt(row[hi], digits) + ']'
def ordered(rows, conditions=CONDITIONS):
    indexed={(r['arm'],r['condition']):r for r in rows}
    return [indexed[(arm,c)] for arm in ARMS for c in conditions]

def numeric_rows():
    result={}
    cells=ordered(rq1('RQ1_CELL_RESULTS'))
    gaps=ordered(rq1('RQ1_GAP_RESULTS'),CONDITIONS[1:])
    result['rq1_battery.tex']=[[fmt(r['accuracy'],3),interval(r,'lower_95%','upper_95%',3)] for r in cells]+[[fmt(r['source_minus_condition_gap'],3),interval(r,'lower_95%','upper_95%',3)] for r in gaps]
    result['rq1_sufficiency.tex']=[[fmt(r['contrast_description_minus_no_description']),interval(r,'lower_95%','upper_95%')] for r in rq1('RQ1_PLANNED_CONTRASTS') if r['contrast_family']=='description_vs_no_description_sufficiency_gap']
    trans=ordered(rq1('RQ1_TRANSITION_RESULTS'),CONDITIONS[1:]); ix={(r['arm'],r['condition']):r for r in trans}
    result['rq1_transitions_full.tex']=[[integer(r['C_to_W']),fmt(r['C_to_W/source_correct'],3),integer(r['W_to_C']),fmt(r['W_to_C/source_wrong'],3),fmt(r['W_to_C/N'],3),integer(r['churn_count']),integer(r['net_movement_W_to_C_minus_C_to_W'])] for r in trans]
    def countpct(r,n,d):return f"{integer(r[n])} ({100*int(r[n])/int(r[d]):.2f}\\%)"
    result['rq1_transitions_compact.tex']=[[countpct(ix[(a,'grey')],'C_to_W','source_correct'),countpct(ix[(a,'grey')],'W_to_C','source_wrong')] for a in ARMS]+[[countpct(ix[(a,'mismatch')],'C_to_W','source_correct'),countpct(ix[(a,'mismatch')],'churn_count','N'),countpct(ix[(a,'mirror')],'churn_count','N')] for a in ARMS]
    result['rq2_drift.tex']=[[fmt(r['mean_raw_drift']),interval(r,'ci_lower_95','ci_upper_95'),integer(r['valid_pair_n'])] for r in ordered(rq2('RQ2_PRIMARY_DRIFT_RESULTS'),CONDITIONS[1:])]
    result['rq2_known_change.tex']=[[integer(r['valid_paired_n']),fmt(r['image_removal_mean_drift']),fmt(r['alternate_answer_mean_drift']),fmt(r['answer_minus_image_mean_difference'],signed=True),interval(r,'difference_ci_lower_95','difference_ci_upper_95')] for r in rq2('RQ2_KNOWN_CHANGE_RESULTS')]
    result['rq2_planned_contrasts.tex']=[[fmt(r['left_mean_drift']),fmt(r['right_mean_drift']),fmt(r['mean_difference']),interval(r,'ci_lower_95','ci_upper_95')] for r in rq2('RQ2_PLANNED_CONTRASTS')]
    coverage=[r for r in read('reports/frozen/rq2_final/audits/RQ2_PARSE_COVERAGE.tsv') if r['family']=='stage2_primary']
    result['rq2_parse_coverage_full.tex']=[[integer(r[k]) for k in ['raw_generation_n','strict_final_success_n','tolerant_final_success_n','source_relative_pairable_final_n']] for r in ordered(coverage)]
    counts={(r['arm'],r['stratum']):r for r in read('provenance/PRED_A_CMC_COUNTS.tsv')}
    result['pred_a_compact.tex']=[];result['pred_a_strata.tex']=[]
    for arm in ARMS:
        r=counts[(arm,'all')];n,u,j=(int(r[k]) for k in ['records','usable_final','judge_correct'])
        result['pred_a_compact.tex'].append([integer(u),f'{u/n*100:.2f}\\%',integer(j),fmt(j/u),fmt(j/n)])
        for st in ['stage1_correct','stage1_wrong']:
            r=counts[(arm,st)];n,u,j=(int(r[k]) for k in ['records','usable_final','judge_correct'])
            result['pred_a_strata.tex'].append([integer(n),integer(u),integer(j),fmt(j/u),fmt(j/n)])
    eligibility=read('provenance/PRED_A_ELIGIBILITY_COUNTS.tsv')
    result['pred_a_coverage.tex']=[[integer(r['empty_final']),integer(r['ambiguous']),integer(int(r['empty_final'])+int(r['ambiguous'])),integer(r['tolerant_recovered']),integer(r['usable_final'])] for r in eligibility]
    units={(r['arm'],r['condition'],r['text_unit']):r for r in read('reports/frozen/rq2_final/secondary/RQ2_TEXT_UNIT_SENSITIVITY.tsv')}
    names=['tolerant_final_primary','tolerant_reasoning_sensitivity','raw_full_generation_sensitivity'];out=[]
    for arm in ARMS:
        for c in CONDITIONS[1:]:
            out.append([integer(units[(arm,c,names[0])]['valid_pair_n'])]+[fmt(units[(arm,c,n)]['mean_raw_drift']) for n in names])
    echo={(r['arm'],r['representation']):r for r in read('reports/frozen/rq2_final/secondary/RQ2_KNOWN_CHANGE_ECHO_SENSITIVITY.tsv')}
    for arm in ARMS:
        for rep in ['final_span','reasoning_only','answer_choice_masked_final']:
            r=echo[(arm,rep)];out.append([fmt(r['image_removal_mean_drift']),fmt(r['alternate_answer_mean_drift']),fmt(r['answer_minus_image_mean_difference'],signed=True)+' '+interval(r,'difference_ci_lower_95','difference_ci_upper_95')])
    result['rq2_text_unit_sensitivity.tex']=out
    return result

def render_tables():
    rendered={}
    for name,rows in numeric_rows().items():
        template=(ROOT/'scripts/table_templates'/name).read_text(encoding='utf-8')
        assert len(re.findall(r'@@ROW_\d+@@',template))==len(rows),name
        for i,values in enumerate(rows):template=template.replace(f'@@ROW_{i}@@',' & '.join(values))
        assert '@@ROW_' not in template,name
        rendered[name]=template
    return rendered

def validate_tables():
    checks=[]
    for name,text in render_tables().items():
        expected=(ROOT/'thesis/Tables/final'/name).read_text(encoding='utf-8')
        checks.append({'path':'thesis/Tables/final/'+name,'matches_approved_bytes':text==expected})
    return checks

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path)
    ap.add_argument('--check',action='store_true')
    args=ap.parse_args()
    if args.output_dir:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        for name,text in render_tables().items():
            destination=args.output_dir/name
            if destination.resolve()==(ROOT/'thesis/Tables/final'/name).resolve():raise SystemExit('Choose a separate generated output directory')
            destination.write_text(text,encoding='utf-8')
    checks=validate_tables();print(json.dumps({'checks':checks,'status':'PASS' if all(c['matches_approved_bytes'] for c in checks) else 'FAIL'},indent=2))
    if not all(c['matches_approved_bytes'] for c in checks):raise SystemExit(1)
if __name__=='__main__':main()
