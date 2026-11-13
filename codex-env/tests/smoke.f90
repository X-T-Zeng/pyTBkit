program smoke
  use omp_lib
  implicit none
  double precision :: a(2,2), w(2), work(10), total
  integer :: info,i,nt
  a=reshape([1d0,0d0,0d0,3d0],[2,2])
  call dsyev('N','U',2,a,2,w,work,10,info)
  if(info/=0.or.maxval(abs(w-[1d0,3d0]))>1d-12) stop 1
  total=0d0
  nt=0
!$omp parallel reduction(+:nt)
  nt=nt+1
!$omp end parallel
!$omp parallel do reduction(+:total)
  do i=1,1000
    total=total+dble(i)
  end do
!$omp end parallel do
  if(abs(total-500500d0)>1d-9.or.nt<2) stop 2
  print *, 'Fortran LAPACK + OpenMP PASS; threads=',nt
end program
