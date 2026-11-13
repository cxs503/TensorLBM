!============================== License GPLv3 ===================================
!    CudaPhaseField is three-component flow solver based on th lattice Boltzmann method which runs on the NVIDIA GPU.
!    Copyright (C) 2021 Amin Zar, aminpopjoury@gmail.com


!    This program is free software: you can redistribute it and/or modify
!    it under the terms of the GNU General Public License as published by
!    the Free Software Foundation, either version 3 of the License, or
!    (at your option) any later version.

!    This program is distributed in the hope that it will be useful,
!    but WITHOUT ANY WARRANTY; without even the implied warranty of
!    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!    GNU General Public License for more details.

!    You should have received a copy of the GNU General Public License
!    along with this program.  If not, see <https://www.gnu.org/licenses/>.
!================================================================================

module device_var
    use precision_m
    implicit none
	integer, constant, parameter :: nxd = 254
	integer, constant, parameter ::	nyd = 254
	real(fp_kind), constant, parameter :: density(3) = [1000._fp_kind, 500.0_fp_kind, 100.0_fp_kind]
    integer, constant, parameter :: ex(0:8) = [0, 1, 0,-1, 0, 1,-1,-1, 1]
    integer, constant, parameter :: ey(0:8) = [0, 0, 1, 0,-1, 1, 1,-1,-1]
    real(fp_kind), constant, parameter :: wa(0:8) = [16,4, 4, 4, 4, 1, 1, 1, 1] / 36._fp_kind
    real(fp_kind), constant,parameter :: w = 4._fp_kind
	real(fp_kind),constant, parameter :: m = 0.01_fp_kind
	real(fp_kind),constant, parameter :: w_c = 1._fp_kind/(0.5_fp_kind + 3._fp_kind*m)
    real(fp_kind),parameter :: tau1  = 0.9_fp_kind
    real(fp_kind),parameter :: tau2  = 0.9_fp_kind
    real(fp_kind),parameter :: tau3  = 0.9_fp_kind
    real(fp_kind),parameter :: sigma(3,3) = reshape((/0.1_fp_kind, 0.1_fp_kind,0.1_fp_kind,0.1_fp_kind,0.1_fp_kind,0.1_fp_kind,0.1_fp_kind,0.1_fp_kind,0.1_fp_kind/), (/3,3/))
    real(fp_kind), constant, parameter :: lambda(3) = [(sigma(2,1) +  &
        sigma(3,1) &
        - sigma(3,2)) &
        , (sigma(2,1) + sigma(3,2) - sigma(3,1)) &
        , (sigma(3,1) + sigma(3,2) - sigma(2,1)) ]
    real(fp_kind), constant, parameter :: lambdat = 3._fp_kind / ((1._fp_kind / lambda(1)) &
        + (1._fp_kind / lambda(2)) + (1._fp_kind / lambda(3)))
    real(fp_kind),parameter :: pi = acos(-1._fp_kind)
    
   
	

	

	real(fp_kind),device, dimension(0:nxd+1,0:nyd+1,0:8) :: h1
	real(fp_kind),device, dimension(0:nxd+1,0:nyd+1,0:8) :: h2
	real(fp_kind),device, dimension(nxd,nyd) :: htemp11, htemp12, htemp13, htemp14, htemp15, htemp16, htemp17, htemp18
	real(fp_kind),device, dimension(nxd,nyd) :: htemp21, htemp22, htemp23, htemp24, htemp25, htemp26, htemp27, htemp28
	real(fp_kind),device, dimension(nxd,nyd) :: hlph10, hlph11, hlph12, hlph13, hlph14, hlph15, hlph16, hlph17, hlph18
	real(fp_kind),device, dimension(nxd,nyd) :: hlph20, hlph21, hlph22, hlph23, hlph24, hlph25, hlph26, hlph27, hlph28
	real(fp_kind),device, dimension(nxd,nyd) :: efh10, efh11, efh12, efh13, efh14, efh15, efh16, efh17, efh18
	real(fp_kind),device, dimension(nxd,nyd) :: efh20, efh21, efh22, efh23, efh24, efh25, efh26, efh27, efh28
	real(fp_kind),device, dimension(nxd,nyd) :: heq10, heq11, heq12, heq13, heq14, heq15, heq16, heq17, heq18
	real(fp_kind),device, dimension(nxd,nyd) :: heq20, heq21, heq22, heq23, heq24, heq25, heq26, heq27, heq28
	real(fp_kind),device, dimension(nxd,nyd) :: dphi1dx, dphi2dx, dphi3dx, dphi1dy, dphi2dy, dphi3dy
	real(fp_kind),device, dimension(0:nxd+1,0:nyd+1) :: phi1, phi2, phi3
	real(fp_kind),device, dimension(nxd,nyd) :: ni1, ni2, ni3, nj1, nj2, nj3, d2phi1, d2phi2, d2phi3, tmp1, tmp2, tmp3
	real(fp_kind),device, dimension(nxd,nyd) :: ga_wa0, ga_wa1, ga_wa2, ga_wa3, ga_wa4, ga_wa5, ga_wa6, ga_wa7, ga_wa8
	real(fp_kind),device, dimension(nxd,nyd) :: gamma0, gamma1, gamma2, gamma3, gamma4, gamma5, gamma6, gamma7, gamma8
	real(fp_kind),device, dimension(nxd,nyd) :: eu0, eu1, eu2, eu3, eu4, eu5, eu6, eu7, eu8, mu1, mu2, mu3
	
	! g(:,:,i), i=0..8: the 9 D2Q9 pressure-population values, same layout rationale as h1/h2.
	real(fp_kind),device, dimension(0:nxd+1,0:nyd+1,0:8) :: g
	real(fp_kind),device, dimension(nxd,nyd) :: gtemp1, gtemp2, gtemp3, gtemp4, gtemp5, gtemp6, gtemp7, gtemp8
	real(fp_kind),device, dimension(nxd,nyd) :: hlpg0, hlpg1, hlpg2, hlpg3, hlpg4, hlpg5, hlpg6, hlpg7, hlpg8
	real(fp_kind),device, dimension(nxd,nyd) :: ef0, ef1, ef2, ef3, ef4, ef5, ef6, ef7, ef8
	real(fp_kind),device, dimension(nxd,nyd) :: geq0, geq1, geq2, geq3, geq4, geq5, geq6, geq7, geq8
	
	real(fp_kind),device, dimension(nxd,nyd) :: p, rho, ux, uy, u2, tauu, heav1, heav2
	real(fp_kind),device, dimension(nxd,nyd) :: fpx, fpy, fsx, fsy, fmx, fmy, fx, fy, sxx, sxy, syy, fg
	integer,device, dimension(0:nxd+1,0:nyd+1) :: is_solid_node
	
	! real(fp_kind),device, allocatable, dimension(:,:) :: h10, h11, h12, h13, h14, h15, h16, h17, h18
	! real(fp_kind),device, allocatable, dimension(:,:) :: h20, h21, h22, h23, h24, h25, h26, h27, h28
	! real(fp_kind),device, allocatable, dimension(:,:) :: htemp11, htemp12, htemp13, htemp14, htemp15, htemp16, htemp17, htemp18
	! real(fp_kind),device, allocatable, dimension(:,:) :: htemp21, htemp22, htemp23, htemp24, htemp25, htemp26, htemp27, htemp28
	! real(fp_kind),device, allocatable, dimension(:,:) :: hlph10, hlph11, hlph12, hlph13, hlph14, hlph15, hlph16, hlph17, hlph18
	! real(fp_kind),device, allocatable, dimension(:,:) :: hlph20, hlph21, hlph22, hlph23, hlph24, hlph25, hlph26, hlph27, hlph28
	! real(fp_kind),device, allocatable, dimension(:,:) :: efh10, efh11, efh12, efh13, efh14, efh15, efh16, efh17, efh18
	! real(fp_kind),device, allocatable, dimension(:,:) :: efh20, efh21, efh22, efh23, efh24, efh25, efh26, efh27, efh28
	! real(fp_kind),device, allocatable, dimension(:,:) :: heq10, heq11, heq12, heq13, heq14, heq15, heq16, heq17, heq18
	! real(fp_kind),device, allocatable, dimension(:,:) :: heq20, heq21, heq22, heq23, heq24, heq25, heq26, heq27, heq28
	! real(fp_kind),device, allocatable, dimension(:,:) :: phi1, phi2, phi3, dphi1dx, dphi2dx, dphi3dx, dphi1dy, dphi2dy, dphi3dy
	! real(fp_kind),device, allocatable, dimension(:,:) :: ni1, ni2, ni3, nj1, nj2, nj3, d2phi1, d2phi2, d2phi3, tmp1, tmp2, tmp3
	! real(fp_kind),device, allocatable, dimension(:,:) :: ga_wa0, ga_wa1, ga_wa2, ga_wa3, ga_wa4, ga_wa5, ga_wa6, ga_wa7, ga_wa8
	! real(fp_kind),device, allocatable, dimension(:,:) :: gamma0, gamma1, gamma2, gamma3, gamma4, gamma5, gamma6, gamma7, gamma8
	! real(fp_kind),device, allocatable, dimension(:,:) :: eu0, eu1, eu2, eu3, eu4, eu5, eu6, eu7, eu8, mu1, mu2, mu3
	
	! real(fp_kind),device, allocatable, dimension(:,:) :: g0, g1, g2, g3, g4, g5, g6, g7, g8
	! real(fp_kind),device, allocatable, dimension(:,:) :: gtemp1, gtemp2, gtemp3, gtemp4, gtemp5, gtemp6, gtemp7, gtemp8
	! real(fp_kind),device, allocatable, dimension(:,:) :: hlpg0, hlpg1, hlpg2, hlpg3, hlpg4, hlpg5, hlpg6, hlpg7, hlpg8
	! real(fp_kind),device, allocatable, dimension(:,:) :: ef0, ef1, ef2, ef3, ef4, ef5, ef6, ef7, ef8
	! real(fp_kind),device, allocatable, dimension(:,:) :: geq0, geq1, geq2, geq3, geq4, geq5, geq6, geq7, geq8
	
	! real(fp_kind),device, allocatable, dimension(:,:) :: p, rho, ux, uy, u2, tauu, heav1, heav2
	! real(fp_kind),device, allocatable, dimension(:,:) :: fpx, fpy, fsx, fsy, fmx, fmy, fx, fy, sxx, sxy, syy
	! integer,device, allocatable, dimension(:,:) :: is_solid_node
	
	

	
end module device_var