"""Fetch the reviewed official source, then construct a separately patched copy.

Never resolve 'latest' during tests and never mutate the pristine reference.
Network access is needed only when the pinned commit/fixture blobs are absent.
"""
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess


def prepare(root=None):
    root=Path(root) if root else Path(__file__).resolve().parents[2]
    config=json.loads((root/'python/upstream.json').read_text())
    pristine=root/config['checkout'];patched=root/config['patched_checkout']
    command=['git']
    # Conda git ships its own CA set, which lacks the managed Cloud proxy CA.
    # Keep TLS verification enabled and use the host's trusted CA bundle.
    ca=os.environ.get('GIT_SSL_CAINFO')
    if not ca and Path('/etc/ssl/certs/ca-certificates.crt').exists():ca='/etc/ssl/certs/ca-certificates.crt'
    if ca:command+=['-c','http.sslCAInfo='+ca]
    def git(*args,**kwargs):
        return subprocess.run([*command,'-C',str(pristine),*args],check=True,**kwargs)
    if not (pristine/'.git').exists():
        pristine.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run([*command,'clone','--depth','1','--filter=blob:none','--sparse',config['repository'],str(pristine)],check=True)
    url=subprocess.check_output(['git','-C',str(pristine),'remote','get-url','origin'],text=True).strip()
    if url.rstrip('/')!=config['repository'].rstrip('/'):raise RuntimeError('reference remote does not match the pinned official repository')
    if subprocess.run(['git','-C',str(pristine),'cat-file','-e',config['commit']+'^{commit}'],capture_output=True).returncode:
        git('fetch','--depth','1','origin',config['commit'])
    # Preserve any unexpected local edits instead of checking them away.
    if subprocess.check_output(['git','-C',str(pristine),'diff','--name-only','HEAD'],text=True).strip():
        raise RuntimeError('pristine official reference has local changes; preserve them before updating')
    patterns=['/src/','/LICENSE','/README.md']+['/'+x for x in config['fixtures']]
    git('sparse-checkout','set','--no-cone','--stdin',input='\n'.join(patterns)+'\n',text=True)
    git('checkout','--detach',config['commit'])
    actual=subprocess.check_output(['git','-C',str(pristine),'rev-parse','HEAD'],text=True).strip()
    if actual!=config['commit']:raise RuntimeError('reference revision mismatch')
    patched.mkdir(parents=True,exist_ok=True)
    # This is a generated test reference, never the development working tree.
    if (patched/'src').exists():shutil.rmtree(patched/'src')
    shutil.copytree(pristine/'src',patched/'src')
    if not (patched/'.git').exists():subprocess.run(['git','init','-q',str(patched)],check=True)
    patch=root/config['patch']
    if patch.exists():
        subprocess.run(['git','-C',str(patched),'apply','--check',str(patch)],check=True)
        subprocess.run(['git','-C',str(patched),'apply',str(patch)],check=True)
    metadata=dict(config)
    metadata['patch_sha256']=hashlib.sha256(patch.read_bytes()).hexdigest() if patch.exists() else None
    (patched/'reference.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'repository':config['repository'],'commit':actual,'pristine':str(pristine),'patched':str(patched)},indent=2))
    return pristine,patched

if __name__=='__main__':prepare()
