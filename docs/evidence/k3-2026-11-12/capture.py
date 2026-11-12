from pathlib import Path
import subprocess,os,json
p=Path('/tmp/k3-rootcause-20261112')
a=json.loads((p/'matrix.json').read_text())['base_args']
e={k:v for k,v in os.environ.items() if not k.startswith(('LD_','OPENBLAS_','GOTO_','OMP_','SPACEMIT_EP_','OPENCV_'))};e['OPENBLAS_NUM_THREADS']='8';e['YOLOOP_ORT_DEBUG_ADDR']='1'
cmds=['set pagination off','set print thread-events off','set disable-randomization off','run','bt 8','p $_siginfo','info registers','p $vl','p $vtype','p $vlenb','x/32i $pc-64','info proc mappings','thread apply all bt 4']
args=['gdb','-q','-nx','-batch']
for c in cmds:args+=['-ex',c]
args+=['--args','taskset','-c','0-7',str(p/'bench-system'),*a]
for i in range(1,5):
 r=subprocess.run(args,env=e,cwd=p,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
 text=r.stdout.decode(errors='replace');(p/f'gdb-{i}.txt').write_text(text)
 print('GDB',i,'SIGSEGV' in text,flush=True)
 if 'SIGSEGV' in text:
  print(text[:12500],flush=True)
  break
