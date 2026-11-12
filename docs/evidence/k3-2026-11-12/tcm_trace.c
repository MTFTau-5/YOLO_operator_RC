#define _GNU_SOURCE
#include <sys/mman.h>
#include <unistd.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdint.h>
static void *(*next_mmap)(void *,size_t,int,int,int,off_t);
void *mmap(void *addr,size_t len,int prot,int flags,int fd,off_t off) {
 if (!next_mmap) next_mmap=dlsym(RTLD_NEXT,"mmap");
 char path[64],target[256];
 if (fd>=0 && len==0x60000 && !(flags & MAP_FIXED)) {
  snprintf(path,sizeof path,"/proc/self/fd/%d",fd);
  ssize_t n=readlink(path,target,sizeof target-1);
  if (n>0) {
   target[n]=0;
   if (!strcmp(target,"/dev/tcm")) {
    void *mapped=next_mmap(addr,len,prot,flags,fd,off);
    char msg[256];int count=snprintf(msg,sizeof msg,"TCM_TRACE start=%p end=%p len=%zu offset=%ld\n",mapped,(char*)mapped+len,len,(long)off);write(2,msg,count);
    return mapped;
   }
  }
 }
 return next_mmap(addr,len,prot,flags,fd,off);
}
