#include <fftw3.h>
#include <math.h>
#include <stdio.h>
int main(void) {
  fftw_complex x[4]={{1,0},{1,0},{1,0},{1,0}}, y[4];
  fftw_plan p=fftw_plan_dft_1d(4,x,y,FFTW_FORWARD,FFTW_ESTIMATE);
  if(!p) return 1;
  fftw_execute(p); fftw_destroy_plan(p);
  if(fabs(y[0][0]-4)>1e-12) return 2;
  for(int i=1;i<4;i++) if(hypot(y[i][0],y[i][1])>1e-12) return 3;
  puts("C FFTW PASS"); return 0;
}
