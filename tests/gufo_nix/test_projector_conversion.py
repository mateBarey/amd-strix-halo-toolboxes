import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import gguf

SCRIPT = Path(__file__).resolve().parents[2] / 'toolboxes/gufo-nix/convert_gufo_projector.py'


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while chunk := source.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def schema():
    result = []
    for i in range(27):
        p = f'v.blk.{i}.'
        for n in ('ln1', 'ln2'):
            for s in ('weight', 'bias'): result.append((p+n+'.'+s, (1152,), False))
        for n, a, b in [('attn_qkv',1152,3456), ('attn_out',1152,1152), ('ffn_up',1152,4304), ('ffn_down',4304,1152)]:
            result.append((p+n+'.weight',(a,b),True))
            result.append((p+n+'.bias',(b,),False))
    result += [('v.patch_embd.weight',(16,16,3,1152),False),('v.patch_embd.weight.1',(16,16,3,1152),False),('v.patch_embd.bias',(1152,),False),('v.position_embd.weight',(1152,2304),False),('v.post_ln.weight',(1152,),False),('v.post_ln.bias',(1152,),False),('mm.0.weight',(4608,4608),True),('mm.0.bias',(4608,),False),('mm.2.weight',(4608,5120),True),('mm.2.bias',(5120,),False)]
    return result


def fixture(path, bad_dtype=False):
    w = gguf.GGUFWriter(path, 'clip')
    w.add_custom_alignment(64)
    fields = {'general.type':'mmproj', 'general.name':'synthetic projector', 'clip.projector_type':'qwen3vl_merger'}
    for k,v in fields.items(): w.add_string(k,v)
    for k,v in {'projection_dim':5120,'patch_size':16,'embedding_length':1152,'feed_forward_length':4304,'block_count':27,'attention.head_count':16,'spatial_merge_size':2,'image_size':768}.items(): w.add_uint32('clip.vision.'+k,v)
    w.add_bool('clip.has_vision_encoder',True)
    w.add_bool('clip.use_gelu',True)
    w.add_float32('clip.vision.attention.layer_norm_epsilon',1e-6)
    for k in ('image_mean','image_std'): w.add_array('clip.vision.'+k,[0.5]*3)
    w.add_array('clip.vision.is_deepstack_layers',[False]*27)
    w.add_uint32('general.file_type',1)
    w.add_key_value('test.u8',255,gguf.GGUFValueType.UINT8)
    w.add_key_value('test.i16',-123,gguf.GGUFValueType.INT16)
    w.add_key_value('test.strings',['alpha','βeta'],gguf.GGUFValueType.ARRAY,gguf.GGUFValueType.STRING)
    w.add_key_value('test.ints',[-1,2],gguf.GGUFValueType.ARRAY,gguf.GGUFValueType.INT16)
    for name,shape,matrix in schema():
        dtype=np.dtype('float16' if matrix or (bad_dtype and name=='v.post_ln.weight') else 'float32')
        w.add_tensor_info(name,tuple(reversed(shape)),dtype,int(np.prod(shape))*dtype.itemsize)
    arch=w.kv_data[0].pop('general.architecture')
    w.kv_data[0]['general.architecture']=arch  # Not first: conversion must preserve order.
    total=sum(gguf.GGUFWriter.ggml_pad(t.nbytes,64) for t in w.tensors[0].values())
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    fp=w.fout[0]
    start=gguf.GGUFWriter.ggml_pad(fp.tell(),64)
    fp.truncate(start+total)
    w.close()
    r=gguf.GGUFReader(path,'r+')
    samples=np.array([0.0,-0.0,1.0,-2.0,1+1/256,1+3/256,np.inf,-np.inf,np.nan,2**-24],dtype=np.float16)
    for t in r.tensors:
        if t.tensor_type==gguf.GGMLQuantizationType.F16: t.data.reshape(-1)[:len(samples)]=samples
        else: t.data.reshape(-1)[:3]=np.array([0.25,-1.5,42.0],dtype=np.float32)
    r.data.flush()


class ConversionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='gufo-projector-test-')
        cls.root=Path(cls.tmp.name)
        cls.source=cls.root/'input.gguf'
        fixture(cls.source)
    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()
    def run_cli(self,*args):
        return subprocess.run([sys.executable,str(SCRIPT),*map(str,args)],text=True,capture_output=True)
    def test_roundtrip_exact_contract(self):
        self.assertTrue(SCRIPT.is_file(),'F16-to-BF16 projector conversion command is missing')
        output=self.root/'converted.gguf'
        before=file_sha256(self.source)
        p=self.run_cli(self.source,output)
        self.assertEqual(p.returncode,0,p.stdout+p.stderr)
        print(p.stdout,end='')
        import resource
        print('Conversion max RSS KiB:', resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        self.assertEqual(before,file_sha256(self.source))
        a,b=gguf.GGUFReader(self.source),gguf.GGUFReader(output)
        self.assertEqual(a.alignment,b.alignment)
        self.assertEqual(a.endianess,b.endianess)
        self.assertEqual(list(a.fields),list(b.fields))
        for key in a.fields:
            if key.startswith('GGUF.'): continue
            self.assertEqual(a.fields[key].types,b.fields[key].types,key)
            expected=32 if key=='general.file_type' else a.fields[key].contents()
            self.assertEqual(expected,b.fields[key].contents(),key)
        self.assertEqual(len(b.tensors),334)
        counts={gguf.GGMLQuantizationType.F32:0,gguf.GGMLQuantizationType.BF16:0}
        expected_bits=np.array([0,0x8000,0x3f80,0xc000,0x3f80,0x3f82,0x7f80,0xff80,0x7fc0,0x3380],dtype=np.uint16)
        for x,y in zip(a.tensors,b.tensors):
            self.assertEqual(x.name,y.name)
            np.testing.assert_array_equal(x.shape,y.shape)
            counts[y.tensor_type]+=1
            if x.tensor_type==gguf.GGMLQuantizationType.F16:
                self.assertEqual(y.tensor_type,gguf.GGMLQuantizationType.BF16)
                np.testing.assert_array_equal(y.data.view(np.uint16).reshape(-1)[:10],expected_bits)
                self.assertEqual(np.count_nonzero(y.data.reshape(-1)[20:]),0)
            else:
                np.testing.assert_array_equal(x.data.view(np.uint8),y.data.view(np.uint8))
        self.assertEqual(list(counts.values()),[224,110])
        p=self.run_cli(output,'--check-only')
        self.assertEqual(p.returncode,0,p.stderr)
        print(p.stdout,end='')
        size=output.stat().st_size
        p=self.run_cli(self.source,output)
        self.assertNotEqual(p.returncode,0)
        self.assertIn('exists',p.stderr)
        self.assertEqual(output.stat().st_size,size)
        link=self.root/'dangling-output.gguf'
        link.symlink_to(self.root/'missing-target.gguf')
        p=self.run_cli(self.source,link)
        self.assertNotEqual(p.returncode,0)
        self.assertTrue(link.is_symlink())
        self.assertFalse((self.root/'missing-target.gguf').exists())
        p=self.run_cli(self.source,'--check-only')
        self.assertNotEqual(p.returncode,0)
        p=self.run_cli(self.source,self.source)
        self.assertNotEqual(p.returncode,0)
    def test_bad_dtype_rejected_without_output(self):
        self.assertTrue(SCRIPT.is_file(),'F16-to-BF16 projector conversion command is missing')
        source=self.root/'bad.gguf'; fixture(source,bad_dtype=True)
        output=self.root/'bad-output.gguf'
        p=self.run_cli(source,output)
        self.assertNotEqual(p.returncode,0)
        self.assertIn('v.post_ln.weight',p.stderr)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob('*.partial.gguf')))

if __name__=='__main__': unittest.main(verbosity=2)
