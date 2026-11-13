"""Reproducible fresh bootstrap and refusal to overwrite a modified reference."""
import importlib.util
import json
from pathlib import Path
import subprocess
import pytest


def test_fresh_reference_bootstrap_and_preserve_edits(tmp_path):
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('prepare_reference',root/'python/tools/prepare_upstream.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    remote=tmp_path/'remote';remote.mkdir()
    def git(*args):return subprocess.check_output(['git','-C',str(remote),*args],text=True).strip()
    git('init','-q');(remote/'src').mkdir();(remote/'src/model.m').write_text('x=1;\n')
    (remote/'LICENSE').write_text('fixture\n');(remote/'README.md').write_text('fixture\n')
    git('add','.');git('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixture')
    commit=git('rev-parse','HEAD');project=tmp_path/'project';(project/'python').mkdir(parents=True)
    config={'repository':str(remote),'commit':commit,'checkout':'work/reference','patched_checkout':'work/fixed','patch':'python/absent.patch','fixtures':[]}
    (project/'python/upstream.json').write_text(json.dumps(config))
    reference,fixed=module.prepare(project)
    assert (fixed/'src/model.m').read_text()=='x=1;\n'
    module.prepare(project) # idempotent, no false dirty-tree error
    (reference/'src/model.m').write_text('x=2;\n')
    with pytest.raises(RuntimeError,match='local changes'):module.prepare(project)
    assert (reference/'src/model.m').read_text()=='x=2;\n'
