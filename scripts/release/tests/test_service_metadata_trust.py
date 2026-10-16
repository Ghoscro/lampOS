"""Run release shell scripts against a filesystem-only fake GCS transport."""
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

RELEASE = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('jq'), 'release scripts require jq')
class MetadataTrustTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bindir = self.root / 'bin'; self.bindir.mkdir()
        tool = self.bindir / 'gsutil'
        tool.write_text(f'#!{sys.executable}\n' + '''import os, pathlib, shutil, sys
root=pathlib.Path(os.environ['MOCK_GCS'])
args=sys.argv[1:]; i=args.index('cp'); source,dest=args[i+1:i+3]
if source.startswith('gs://'):
    shutil.copyfile(root/'metadata.json',dest)
else:
    with (root/'writes').open('a') as stream: stream.write(dest+'\\n')
    shutil.copyfile(source, root/('metadata.json' if dest.endswith('metadata.json') else 'artifact.zip'))
''')
        tool.chmod(0o755)
        self.metadata = self.root / 'metadata.json'
        self.metadata.write_text(json.dumps({'hal': {'version':'1.0.0'}, 'lamp-os': {
            'version':'1.0.0','service':'lamp-os','replaces':['hal','os-server']}}))
        self.archive = self.root / 'package.zip'
        with zipfile.ZipFile(self.archive, 'w') as archive:
            archive.writestr('VERSION','1.2.3')
            info=zipfile.ZipInfo('bin/service'); info.external_attr=0o100755<<16
            archive.writestr(info,'#!/bin/sh\nexit 0\n')
        self.env = {k:v for k,v in os.environ.items() if not k.startswith('OTA_')}
        self.env.update(PATH=str(self.bindir)+':'+os.environ['PATH'], MOCK_GCS=str(self.root),
                        COMPONENT='lamp-os',V='1.2.3',PACKAGE=str(self.archive),REPLACES='hal os-server',
                        FOR='lamp',ACTIVATE='1')

    def run_script(self, name):
        args = ['lamp-os'] if name == 'promote-ota.sh' else []
        return subprocess.run(['bash',str(RELEASE/name),*args],env=self.env,capture_output=True,text=True)

    def test_unsigned_upload_preserves_hash_and_then_promotes(self):
        result=self.run_script('upload-service.sh')
        self.assertEqual(result.returncode,0,result.stderr)
        data=json.loads(self.metadata.read_text())
        self.assertNotIn('signed',data)
        self.assertEqual(data['lamp-os']['sha256'],hashlib.sha256(self.archive.read_bytes()).hexdigest())
        self.assertNotIn('activate_by_device',data['lamp-os'])
        result=self.run_script('promote-ota.sh')
        self.assertEqual(result.returncode,0,result.stderr)
        data=json.loads(self.metadata.read_text())
        self.assertEqual(data['lamp-os']['activate_by_device'],{'lamp':True})
        self.assertEqual(data['lamp-os']['min_version_by_device'],{'lamp':'1.2.3'})

    def test_signed_feed_is_not_downgraded(self):
        payload=base64.b64encode(self.metadata.read_bytes()).decode()
        for signed in ({'signed':{'format':'autonomous-ota/v1','payload':payload,'signature':{'algorithm':'ed25519'}}},
                       {'format':'autonomous-ota/v1','payload':payload,'signature':{'algorithm':'ed25519'}}):
            for script in ('upload-service.sh','promote-ota.sh'):
                self.metadata.write_text(json.dumps(signed))
                result=self.run_script(script)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('refusing unsigned downgrade',result.stderr)
                self.assertFalse((self.root/'writes').exists())

    def test_ordinary_service_promotion_allows_unsigned_but_preserves_signed(self):
        self.env.pop('ACTIVATE')
        result = self.run_script('promote-ota.sh')
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(self.metadata.read_text())
        self.assertEqual(data['lamp-os']['min_version_by_device'], {'lamp': '1.0.0'})
        self.assertNotIn('activate_by_device', data['lamp-os'])
        (self.root / 'writes').unlink()
        payload = base64.b64encode(self.metadata.read_bytes()).decode()
        self.metadata.write_text(json.dumps({'signed': {'format': 'autonomous-ota/v1',
            'payload': payload, 'signature': {'algorithm': 'ed25519'}}}))
        result = self.run_script('promote-ota.sh')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('refusing unsigned downgrade', result.stderr)
        self.assertFalse((self.root / 'writes').exists())

    def test_partial_and_invalid_config_never_publish(self):
        invalid=self.root/'key'; invalid.write_text('not a key')
        for config in ({'OTA_SIGNING_KEY_ID':'only-id'},
                       {'OTA_SIGNING_PRIVATE_KEY':str(invalid)},
                       {'OTA_SIGNING_PRIVATE_KEY':str(invalid),'OTA_SIGNING_KEY_ID':'bad'},
                       {'OTA_METADATA_SIGNED_ONLY':'1'}):
            for script in ('upload-service.sh','promote-ota.sh'):
                with self.subTest(config=config,script=script):
                    self.env.update(config)
                    result=self.run_script(script)
                    self.assertNotEqual(result.returncode,0)
                    self.assertFalse((self.root/'writes').exists())
                    for key in config:self.env.pop(key)


if __name__=='__main__': unittest.main()
