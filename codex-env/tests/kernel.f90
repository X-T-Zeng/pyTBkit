subroutine band(n,k,m,e)
  implicit none
  integer, intent(in) :: n
  double precision, intent(in) :: k(n), m
  double precision, intent(out) :: e(n)
  integer :: i
!$omp parallel do
  do i=1,n
    e(i)=sqrt(sin(k(i))**2+(m+cos(k(i)))**2)
  end do
!$omp end parallel do
end subroutine
