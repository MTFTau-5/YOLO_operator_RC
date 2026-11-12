#include <onnxruntime_cxx_api.h>
#include <spacemit_ort_env.h>
#include <vector>
#include <cstdio>
#include <cmath>
int main(int argc,char**argv){
 try {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING,"split_probe");Ort::SessionOptions opts;opts.SetIntraOpNumThreads(1);opts.SetInterOpNumThreads(1);
  Ort::SessionOptionsSpaceMITEnvInit(opts,{{"SPACEMIT_EP_INTRA_THREAD_NUM","1"},{"SPACEMIT_EP_DUMP_SUBGRAPHS","1"}});
  Ort::Session session(env,argv[1],opts);
  std::vector<float> x(1*4*128*400);for(size_t i=0;i<x.size();i++)x[i]=(float)(i%101);
  int64_t shape[]={1,4,128,400};auto mem=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);auto in=Ort::Value::CreateTensor<float>(mem,x.data(),x.size(),shape,4);
  const char* ins[]={"images"};const char* outs[]={"q","k","output0"};auto y=session.Run(Ort::RunOptions{},ins,&in,1,outs,3);
  float error=0;int starts[]={0,32,64},widths[]={32,32,64};
  for(int k=0;k<3;k++){const float*v=y[k].GetTensorData<float>();for(int c=0;c<4;c++)for(int j=0;j<widths[k]*400;j++)error=std::max(error,std::abs(v[c*widths[k]*400+j]-x[c*128*400+starts[k]*400+j]));}
  printf("PASS max_abs_error=%f\n",error);return error==0?0:2;
 }catch(const std::exception&e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}
}
