module inter
contains

subroutine coulchgs(coords1,qs1,coords2,qs2,d,nc1,nc2,ene)

IMPLICIT None
INTEGER, intent(in) :: nc1,nc2
REAL(8), DIMENSION(nc1), intent(in) :: qs1
REAL(8), DIMENSION(nc1,3), intent(in) :: coords1
REAL(8), DIMENSION(nc2), intent(in) :: qs2
REAL(8), DIMENSION(nc2,3), intent(in) :: coords2
REAL(8), DIMENSION(nc1) :: ene_local
REAL(8), DIMENSION(nc1), intent(out) :: ene
REAL(8), DIMENSION(3) :: rij
REAL(8) :: r,d,fac
INTEGER :: i,j,nthrd,omp_get_num_threads

ene=0.d0

!$omp parallel private(i,j,rij,r,ene_local)
ene_local=0.d0

!$omp do
do i=1,nc1
    do j=1,nc2
        rij=(coords1(i,:)-coords2(j,:))
        r=dsqrt(rij(1)**2.0+rij(2)**2.0+rij(3)**2.0)

        if (r < d) then
            fac=exp(-(r-d)**2/1.5)
        else
            fac=1.0
        end if

        ene_local(i)=ene_local(i)+fac*qs1(i)*qs2(j)/r
    end do
end do
!$omp end do

!$omp critical
ene=ene+ene_local
!$omp end critical
!$omp end parallel

end subroutine coulchgs

subroutine coultdc(trdena,grida,dva,trdend,gridd,dvd,thresh,na,nd,couptd)

IMPLICIT None
! Input Variables
INTEGER, intent(in) :: na,nd
REAL(8), intent(in) :: dva,dvd,thresh
REAL(8), DIMENSION(na), intent(in) :: trdena
REAL(8), DIMENSION(nd), intent(in) :: trdend
REAL(8), DIMENSION(na,3), intent(in) :: grida
REAL(8), DIMENSION(nd,3), intent(in) :: gridd
! Internal and output Variables
REAL(8), DIMENSION(3) :: rij
REAL(8) :: r,couptd_local,width
INTEGER :: i,j,OMP_GET_NUM_THREADS
REAL(8), intent(out) :: couptd

! Description of the Variables:
! rij is the distance between two points of the grid
! couptd is the coupling

couptd = 0.
width = 1. / dsqrt (2*(dva**(2.0/3.0) + dvd**(2.0/3.0)) )

!$omp parallel private(i,j,rij,r,couptd_local)
couptd_local = 0.

!$omp do
do i=1,na
    if (abs(trdena(i)) < thresh) cycle
    do j=1,nd

        if (abs(trdend(j)) < thresh) cycle
        rij=(grida(i,:)-gridd(j,:))
        r=dsqrt(rij(1)**2.0+rij(2)**2.0+rij(3)**2.0)

        couptd_local=couptd_local+trdena(i)*trdend(j)/r * erf(width*r)

    end do
end do
!$omp end do

!$omp critical
couptd=couptd+couptd_local
!$omp end critical

!$omp end parallel

couptd=couptd*dva*dvd

end subroutine coultdc

subroutine getpot(coords1,qs1,coords2,d,nc1,nc2,pot)

IMPLICIT None
INTEGER, intent(in) :: nc1,nc2
REAL(8), DIMENSION(nc1), intent(in) :: qs1
REAL(8), DIMENSION(nc1,3), intent(in) :: coords1
REAL(8), DIMENSION(nc2,3), intent(in) :: coords2
REAL(8), DIMENSION(nc2) :: pot_local
REAL(8), DIMENSION(nc2), intent(out) :: pot
REAL(8), DIMENSION(3) :: rij
REAL(8) :: r,d,fac
INTEGER :: i,j,nthrd,omp_get_num_threads

pot=0.d0

!$omp parallel private(i,j,rij,r,pot_local)
pot_local=0.d0

!$omp do
do i=1,nc1
    do j=1,nc2
        rij=(coords1(i,:)-coords2(j,:))
        r=dsqrt(rij(1)**2.0+rij(2)**2.0+rij(3)**2.0)

        if (r < d) then
            fac=exp(-(r-d)**2/1.5)
        else
            fac=1.0
        end if

        pot_local(j)=pot_local(j)+fac*qs1(i)/r
    end do
end do
!$omp end do

!$omp critical
pot=pot+pot_local
!$omp end critical
!$omp end parallel

end subroutine getpot
end module inter
