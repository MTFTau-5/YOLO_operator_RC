from pathlib import Path
import subprocess,os,json,time,resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
p=Path('/home/epoch/work/k3/YOLO_operator_RC');d=Path('/tmp/k3-rootcause-20261112')
vendor=str(p/'build-k3/app/bench_pipeline');system=str(d/'bench-system')
base=['--backend','spacemit','--model',str(p.parent/'data/best.spacemit.fp16.onnx'),'--in0','images','--heads','output0','--model-type','v8-pose','--num-classes','1','--num-kpts','4','--video',str(p.parent/'data/Screencast from 2026-10-13 20-00-08.webm'),'--frames','1','--serial','--gpu','-1','--threads','8','--cores-a','0-7','--video-first']
cases=[('system_blas8',system,None,'8'),('system_blas1',system,None,'1'),('vendor_no_blas',vendor,None,'8'),('vendor_system_blas8',vendor,'/usr/lib/riscv64-linux-gnu/libopenblas.so.0','8'),('vendor_vendor_blas8',vendor,'/opt/openblas-spacemit/lib/libopenblas.so.0','8'),('vendor_vendor_blas1',vendor,'/opt/openblas-spacemit/lib/libopenblas.so.0','1')]
results=[]
for name,exe,preload,threads in cases:
 env={k:v for k,v in os.environ.items() if not k.startswith(('LD_','OPENBLAS_','GOTO_','OMP_','SPACEMIT_EP_','OPENCV_'))}
 env['OPENBLAS_NUM_THREADS']=threads
 if preload:env['LD_PRELOAD']=preload
 libs=subprocess.run(['ldd',exe],env=env,text=True,capture_output=True).stdout
 (d/(name+'-ldd.txt')).write_text(libs)
 row={'case':name,'exe':exe,'env':{'OPENBLAS_NUM_THREADS':threads,'LD_PRELOAD':preload},'runs':[]}
 for i in range(10):
  start=time.monotonic()
  try:
   r=subprocess.run(['taskset','-c','0-7',exe,*base],env=env,cwd=d,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=25)
   rc=r.returncode;out=r.stdout.decode(errors='replace')
  except subprocess.TimeoutExpired as e:rc=124;out=(e.stdout or b'').decode(errors='replace')
  (d/f'{name}-{i+1}.log').write_text(out)
  row['runs'].append({'i':i+1,'rc':rc,'seconds':round(time.monotonic()-start,3)})
  print(name,i+1,rc,flush=True)
 results.append(row)
 (d/'matrix.json').write_text(json.dumps({'base_args':base,'results':results},indent=2))
 print('RESULT',name,'success',sum(x['rc']==0 for x in row['runs']),'/',len(row['runs']),flush=True)
