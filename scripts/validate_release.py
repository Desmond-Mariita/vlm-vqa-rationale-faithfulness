#!/usr/bin/env python3
"""Offline, fail-closed validation of the explicitly listed release payload."""
from pathlib import Path
import argparse
import ast
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TITLE = 'Evaluating the Faithfulness of Chain-of-Thought Rationales in Vision-Language Models for Visual Question Answering'
VISUAL_SUFFIXES = {'.png','.jpg','.jpeg','.gif','.webp','.bmp','.tif','.tiff','.svg','.pdf','.eps','.ps','.ico','.avif','.heic','.mp4','.mov','.webm'}
FORBIDDEN_SUFFIXES = {'.pt','.pth','.ckpt','.bin','.safetensors','.ttf','.otf','.woff','.woff2','.pfb','.pfa','.zip','.tar','.7z','.sqlite','.db'}
VCR_ASSET_ROOTS = ('thesis/Figures/records/', 'thesis/Figures/w2c/panels/')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def files(root=ROOT):
    for base,dirs,names in os.walk(root,followlinks=False):
        dirs[:]=sorted(d for d in dirs if d not in {'.git','_discovery','.venv','__pycache__','.pytest_cache'})
        for n in sorted(names):yield Path(base)/n

def visual_kind(path):
    with path.open('rb') as f:b=f.read(4096)
    if b.startswith(b'\x89PNG\r\n\x1a\n'):return 'png'
    if b.startswith(b'\xff\xd8\xff'):return 'jpeg'
    if b.startswith((b'GIF87a',b'GIF89a')):return 'gif'
    if b.startswith(b'%PDF'):return 'pdf'
    if b.startswith((b'II*\x00',b'MM\x00*')):return 'tiff'
    if b.startswith(b'BM'):return 'bmp'
    if b[:4]==b'RIFF' and b[8:12]==b'WEBP':return 'webp'
    if re.search(br'<svg(?:\s|>)',b,re.I):return 'svg'
    if path.suffix.lower() in VISUAL_SUFFIXES:return path.suffix.lower()[1:]
    return None

def image_audit(root=ROOT, policy=None):
    if policy is None:policy=json.loads((root/'provenance/image_policy.json').read_text())
    allowed={row['path']:row for row in policy['visual_files']}
    historical_hashes={row['sha256'] for row in policy.get('historical_vcr_files',[])}
    images=[];failures=[]
    if policy.get('expected_tracked_vcr_raster_count',0)!=0:
        failures.append({'reason':'public release must require zero VCR rasters'})
    paths=set(files(root))
    # A forced Git addition must not hide in a normally excluded cache directory.
    # Source archives without .git still receive the full payload traversal.
    if (root/'.git').exists():
        tracked=subprocess.run(['git','-C',str(root),'ls-files','-z'],check=True,capture_output=True,text=True)
        paths.update(root/rel for rel in tracked.stdout.split('\0') if rel and (root/rel).exists())
    vcr=[]
    for path in sorted(paths):
        rel=path.relative_to(root).as_posix()
        if path.is_symlink():failures.append({'path':rel,'reason':'symlink forbidden'});continue
        kind=visual_kind(path)
        if not kind:continue
        row=allowed.get(rel);h=sha(path)
        known_vcr=rel.startswith(VCR_ASSET_ROOTS) or h in historical_hashes or (row and row['classification']=='THESIS_VCR_EXEMPTION')
        category=row['classification'] if row and row['sha256']==h and row['classification'] in {'SYNTHETIC','AUTHOR_CREATED_NON_VCR'} and not known_vcr else 'FORBIDDEN'
        if known_vcr:vcr.append(rel)
        if category=='FORBIDDEN':failures.append({'path':rel,'reason':'VCR payload forbidden or visual outside exact public path/hash allow-list'})
        images.append({'path':rel,'format':kind,'bytes':path.stat().st_size,'sha256':h,'classification':category})
    actual={r['path'] for r in images}
    for rel in allowed:
        if rel not in actual:failures.append({'path':rel,'reason':'expected visual missing'})
    forbidden_vcr=[rel for rel in vcr if not rel.startswith('thesis/Figures/')]
    return {'status':'PASS' if not failures else 'FAIL','method':'Public payload traversal plus all Git-indexed paths, including ignored directories; suffix and binary signature/SVG detection; exact path/hash allow-list. Known VCR paths, historical raster hashes and former exemptions are forbidden. Unknown visuals fail closed. This checks the current payload, not Git history.','visual_files':images,'visual_file_count':len(images),'thesis_vcr_count':len(vcr),'vcr_outside_thesis_figures':forbidden_vcr,'forbidden_vcr_roots_checked':['data/','reports/','provenance/','tests/','docs/','frozen_execution/','data_prep/','root release assets','Git-indexed ignored directories'],'failures':failures}

def secret_scan(root=ROOT):
    patterns={
        'github_token':r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b',
        'hf_token':r'\bhf_[A-Za-z0-9]{30,}\b',
        'openai_token':r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}\b',
        'private_key':r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
        'aws_access_key':r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
        'signed_url':r'https?://[^\s\"<>]+[?&](?:X-Amz-Signature|sig|token)=[^\s\"<>]{12,}',
        'credential_assignment':r'''(?i)(?:api_key|access_token|password|secret_key)\s*=\s*["']([^"'\n]{16,})["']''',
    }
    findings=[];scanned=0
    for p in files(root):
        if p.is_symlink():continue
        rel=p.relative_to(root).as_posix()
        if p.name in {'.env','id_rsa','id_ed25519','.netrc','.bash_history','.zsh_history'}:findings.append({'file':rel,'line':0,'type':'credential_filename','fingerprint':'filename-only'})
        b=p.read_bytes()
        if b.startswith(b'\x1f\x8b'):
            try:b=gzip.decompress(b)
            except Exception:findings.append({'file':rel,'line':0,'type':'unreadable_archive','fingerprint':'n/a'});continue
        try:s=b.decode('utf-8')
        except UnicodeDecodeError:continue
        scanned+=1
        for kind,pattern in patterns.items():
            for match in re.finditer(pattern,s):
                # Documented environment lookups and synthetic format descriptions are not credential values.
                value=match.group(1) if kind=='credential_assignment' else match.group(0)
                if kind=='credential_assignment' and any(x in value for x in ['os.environ','os.getenv','YOUR_','<','${']):continue
                findings.append({'file':rel,'line':s[:match.start()].count('\n')+1,'type':kind,'fingerprint':hashlib.sha256(value.encode()).hexdigest()[:12]})
    return {'status':'PASS' if not findings else 'FAIL','method':'Transparent Python regex fallback over all UTF-8 payload files and decompressed numeric gzip; filename screening; no credential values retained. Binary visuals are separately hash allow-listed. This is not exhaustive secret-format detection.','available_external_scanners':{x:bool(shutil.which(x)) for x in ['gitleaks','trufflehog']},'text_files_scanned':scanned,'findings':findings}

def thesis_status(root=ROOT):
    p=json.loads((root/'provenance/exact_pdf.json').read_text())
    holds=['thesis/final_thesis.pdf','thesis/Figures/signature.png','thesis/Figures/uni_potsdam_logo.png']
    graph=json.loads((root/'provenance/thesis_dependencies.json').read_text())
    holds=list(dict.fromkeys(holds+[row['path'] for row in graph['files'] if row['withheld']]))
    failures=[]
    for row in graph['files']:
        if row['withheld']:continue
        path=root/row['path']
        if not path.is_file() or sha(path)!=row['sha256']:failures.append(row['path'])
    ok=p['registered_title']==TITLE and p['payload_included'] is False and all(not (root/x).exists() for x in holds) and not failures
    return {'status':'PASS' if ok else 'FAIL','exact_build':'ON_HOLD','withheld_paths':holds,'active_source_hash_failures':failures,'approved_pdf_sha256':p['approved_pdf_sha256'],'pdf_rebuilt':False}

def validate(root=ROOT):
    manifest=json.loads((root/'provenance/release_manifest.json').read_text())
    expected={r['path']:r for r in manifest['files']};failures=[]
    actual={p.relative_to(root).as_posix():p for p in files(root)}
    # Generated numeric outputs are ignored local products; every candidate input is explicit.
    actual={k:p for k,p in actual.items() if not k.startswith('reports/generated/')}
    expected_paths=set(expected)|{'provenance/release_manifest.json'}
    if set(actual)!=expected_paths:failures.append({'kind':'manifest_file_set','missing':sorted(expected_paths-set(actual)),'unexpected':sorted(set(actual)-expected_paths)})
    for rel,row in expected.items():
        p=root/rel
        if not p.is_file() or p.is_symlink():failures.append({'path':rel,'kind':'missing_or_symlink'});continue
        if sha(p)!=row['sha256'] or p.stat().st_size!=row['bytes']:failures.append({'path':rel,'kind':'hash_or_size'})
        if p.stat().st_size>100_000_000:failures.append({'path':rel,'kind':'oversize'})
        if p.suffix.lower() in FORBIDDEN_SUFFIXES:failures.append({'path':rel,'kind':'forbidden_binary_or_font'})
        if p.suffix=='.py':
            try:ast.parse(p.read_text())
            except SyntaxError:failures.append({'path':rel,'kind':'python_syntax'})
        if p.suffix in {'.jsonl'}:failures.append({'path':rel,'kind':'raw_jsonl_not_permitted'})
        if p.suffix in {'.py','.md','.yaml','.yml','.json','.tsv','.csv','.txt','.cff','.tex'}:
            s=p.read_text()
            if re.search(r'/(?:home|Users)/[^/\s]+/',s):failures.append({'path':rel,'kind':'private_absolute_path'})
            if re.search(r'GPU-[a-f0-9-]{30,}',s):failures.append({'path':rel,'kind':'private_device_uuid'})
    image=image_audit(root);secrets=secret_scan(root);thesis=thesis_status(root)
    if image['status']!='PASS':failures.append({'kind':'image_policy','details':image['failures']})
    if secrets['status']!='PASS':failures.append({'kind':'secret_scan','details':secrets['findings']})
    if thesis['status']!='PASS':failures.append({'kind':'thesis_hold_or_identity'})
    return {'status':'PASS' if not failures else 'FAIL','files':len(actual),'bytes':sum(p.stat().st_size for p in actual.values()),'failures':failures,'image_policy':image,'secret_scan':secrets,'thesis':thesis,'scope':'CPU release integrity; no claim of exact PDF build, full GPU experiment reproduction, licence grant or publication approval'}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--thesis-status',action='store_true');ap.add_argument('--image-only',action='store_true');args=ap.parse_args()
    result=thesis_status() if args.thesis_status else image_audit() if args.image_only else validate()
    print(json.dumps(result,indent=2))
    if result['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
