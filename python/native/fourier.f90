subroutine fourier(k, vectors, hopping, output, nk, dim, nr, n)
  implicit none
  integer, intent(in) :: nk, dim, nr, n
  double precision, intent(in) :: k(nk,dim), vectors(nr,dim)
  complex(kind=8), intent(in) :: hopping(nr,n,n)
  complex(kind=8), intent(out) :: output(nk,n,n)
  integer :: q,r,a,i,j
  double precision :: angle
  complex(kind=8) :: phase
  output=(0d0,0d0)
!$omp parallel do private(q,r,a,i,j,angle,phase)
  do q=1,nk
    do r=1,nr
      angle=0d0
      do a=1,dim
        angle=angle+k(q,a)*vectors(r,a)
      end do
      phase=cmplx(cos(angle),sin(angle),kind=8)
      do j=1,n
        do i=1,n
          output(q,i,j)=output(q,i,j)+phase*hopping(r,i,j)
        end do
      end do
    end do
  end do
!$omp end parallel do
end subroutine

subroutine gradient(k, vectors, hopping, output, nk, dim, nr, n)
  implicit none
  integer, intent(in) :: nk, dim, nr, n
  double precision, intent(in) :: k(nk,dim), vectors(nr,dim)
  complex(kind=8), intent(in) :: hopping(nr,n,n)
  complex(kind=8), intent(out) :: output(nk,dim,n,n)
  integer :: q,r,a,b,i,j
  double precision :: angle
  complex(kind=8) :: phase
  output=(0d0,0d0)
!$omp parallel do private(q,r,a,b,i,j,angle,phase)
  do q=1,nk
    do r=1,nr
      angle=0d0
      do a=1,dim
        angle=angle+k(q,a)*vectors(r,a)
      end do
      phase=cmplx(-sin(angle),cos(angle),kind=8)
      do j=1,n
        do i=1,n
          do b=1,dim
            output(q,b,i,j)=output(q,b,i,j)+vectors(r,b)*phase*hopping(r,i,j)
          end do
        end do
      end do
    end do
  end do
!$omp end parallel do
end subroutine
