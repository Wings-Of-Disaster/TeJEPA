#!/usr/bin/env python3
"""Compile every released theorem and emit a portable, path-free certificate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
FILES=['DistillationLoss.lean','SpectralDistillation.lean','ProjectorAnchor.lean','JointAnchoring.lean','TeacherPrediction.lean']
MATHLIB='c44e0c8ee63ca166450922a373c7409c5d26b00b'
STANDARD={'propext','Classical.choice','Quot.sound'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lean',type=Path,help='Optional installed Lean binary; use with --packages')
    parser.add_argument('--packages',type=Path,help='Optional existing mathlib/dependency cache')
    parser.add_argument('--output',type=Path,default=ROOT/'formal/verification.json')
    args=parser.parse_args()
    if bool(args.lean)!=bool(args.packages):parser.error('--lean and --packages must be supplied together')
    env=os.environ.copy()
    if args.lean:
        packages=args.packages.resolve()
        paths=[p/'.lake/build/lib/lean' for p in packages.iterdir() if p.is_dir()]
        paths.extend(p/'.lake/build/lib/lean' for p in (packages/'mathlib/.lake/packages').glob('*') if p.is_dir())
        env['LEAN_PATH']=':'.join(str(p) for p in paths if p.exists())
        command=[str(args.lean.resolve())];mathlib=packages/'mathlib'
    else:
        command=['lake','env','lean'];mathlib=ROOT/'.lake/packages/mathlib'
    revision=subprocess.check_output(['git','-C',str(mathlib),'rev-parse','HEAD'],text=True).strip()
    if revision!=MATHLIB:raise SystemExit('Pinned mathlib revision does not match')
    version=subprocess.check_output(command+['--version'],env=env,cwd=ROOT,text=True)
    if 'version 4.19.0' not in version:raise SystemExit('Use the pinned Lean 4.19.0 toolchain')
    results=[]
    for name in FILES:
        source=ROOT/'formal'/name;text=source.read_text()
        # These files have no nested block comments. All printed theorem axiom
        # dependencies are checked independently of this source-level gate.
        code=re.sub(r'/\-.*?\-/','',text,flags=re.S)
        code=re.sub(r'--[^\n]*','',code)
        if re.search(r'\b(sorry|admit|axiom)\b',code):raise SystemExit('Unproved declaration in '+name)
        result=subprocess.run(command+['formal/'+name],env=env,cwd=ROOT,capture_output=True,text=True)
        log=result.stdout+result.stderr
        if result.returncode or 'sorryAx' in log:
            print(log);raise SystemExit('Proof check failed: '+name)
        declarations=re.findall(r'^theorem\s+(\w+)',text,re.M)
        printed=re.findall(r"'([^']+)' depends on axioms: \[([^\]]*)\]",log)
        printed.extend((name,'') for name in re.findall(r"'([^']+)' does not depend on any axioms",log))
        if {p[0].rsplit('.',1)[-1] for p in printed}!=set(declarations):
            raise SystemExit('Every theorem must print its axiom dependencies: '+name)
        axioms={a.strip() for _,raw in printed for a in raw.split(',') if a.strip()}
        if not axioms<=STANDARD:raise SystemExit('Unexpected axiom dependency: '+name)
        results.append(dict(file='formal/'+name,sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            declarations=declarations,standard_axioms=sorted(axioms),status='passed'))
        print(name+': passed ('+str(len(declarations))+' declarations)',flush=True)
    report=dict(status='passed',lean_version='4.19.0',mathlib_revision=MATHLIB,
        checked_declarations=sum(len(r['declarations']) for r in results),files=results,
        proof_scope='See the theorem-to-claim table in README.md; analytical and conditional results are identified separately.',
        not_certified=['Population characteristic-function identification','Matrix square-root Frechet bounds',
            'Finite numerical square-root iteration','Optimizer dynamics','Downstream accuracy'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('All released proofs passed; '+str(report['checked_declarations'])+' declarations.')


if __name__=='__main__':main()
