import gdb,struct,json
sym='_ZN4mlir5speir9spekernel4ime217spe_direct_conv2dIDF16_DF16_DF16_NS1_25spe_kernel_block_m16n32k8ELm1ELm3ELm3ELm1ELm1ELm1ELm1ELm136EEEvPvlRKSt5arrayIlLm4EES5_lS9_S5_lRKS6_IlLm3EElllllllS5_'
last={};calls=0
def val(reg):return int(gdb.parse_and_eval('$'+reg))
def longs(addr,n):return list(struct.unpack('<'+'q'*n,gdb.selected_inferior().read_memory(addr,8*n)))
class Entry(gdb.Breakpoint):
 def stop(self):
  global calls
  calls+=1
  r={x:val(x) for x in ['pc','sp','a0','a1','a2','a3','a4','a5','a6','a7']}
  r['input_strides']=longs(r['a2'],4);r['weight_strides']=longs(r['a5'],4);r['stack_args']=longs(r['sp'],9)
  r['output_strides']=longs(r['stack_args'][0],3)
  last[gdb.selected_thread().ptid[1]]=r
  return False
gdb.execute('set breakpoint pending on');Entry("*"+sym)
def stopped(event):
 if isinstance(event,gdb.SignalEvent) and event.stop_signal=='SIGSEGV':
  tid=gdb.selected_thread().ptid[1]
  print('ENTRY_CALLS',calls,'FAULT_TID',tid,'LAST_ENTRY',json.dumps(last.get(tid)))
  print('FAULT_REGS',json.dumps({x:val(x) for x in ['pc','sp','a1','a2','a3','a4','s11','s8']}))
  gdb.execute('bt 5');gdb.execute('p $_siginfo');gdb.execute('info proc mappings')
gdb.events.stop.connect(stopped)
