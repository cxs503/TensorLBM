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

module global_subroutines
contains
	attributes(global) subroutine phi_cal
		use precision_m
		use device_var, only : nxd, nyd, is_solid_node, density, phi1, phi2, phi3, rho, h1, h2
		implicit none
		
		integer :: x, y, n(2)
		x = (blockIdx%x-1)*blockDim%x + threadIdx%x
		y = (blockIdx%y-1)*blockDim%y + threadIdx%y
		n(1) = nxd!size(rho,1) 
		n(2) = nyd!size(rho,2) 
		
		if (x<=n(1) .and. x>=1 .and. y>=1 .and.  y<=n(2)) then
			if (is_solid_node(x,y)==0) then
				
					
					
				phi1(x,y)= h1(x,y,0) + h1(x,y,1) + h1(x,y,2) + h1(x,y,3) + h1(x,y,4) + h1(x,y,5) + h1(x,y,6)+ h1(x,y,7)+ h1(x,y,8)
				phi2(x,y)= h2(x,y,0) + h2(x,y,1) + h2(x,y,2) + h2(x,y,3) + h2(x,y,4) + h2(x,y,5) + h2(x,y,6)+ h2(x,y,7)+ h2(x,y,8)
				
				!call syncthreads()
					
				phi3(x,y) = 1._fp_kind - phi1(x,y) - phi2(x,y)
				
				!call syncthreads()
				
				rho(x,y) = phi1(x,y) * density(1) + phi2(x,y) * density(2) + phi3(x,y) * density(3)
	
							
			end if
		end if
	end
	
	attributes(global) subroutine periodic_phi
		use precision_m
		use device_var, only : nxd, nyd, phi1, phi2, phi3
		implicit none
		
		integer :: x, y, n(2)
		x = (blockIdx%x-1)*blockDim%x + threadIdx%x 
		y = (blockIdx%y-1)*blockDim%y + threadIdx%y 
		n(1) = nxd+1
		n(2) = nyd+1
		if (x<=n(1) .and. y<=n(2)) then
		!if (x<=n(1) .and. x>=1 .and. y>=1 .and.  y<=n(2)) then
		
			
		!-- bottom and top boundaries
			phi1(x,0) = phi1(x,nyd)  
			phi1(x,nyd+1) = phi1(x,1)
		!-- left and right boundaries
			phi1(0,y) = phi1(nxd,y)   
			phi1(nxd+1,y) = phi1(1,y) 
			
		!-- bottom and top boundaries
			phi2(x,0) = phi2(x,nyd)  
			phi2(x,nyd+1) = phi2(x,1)
		!-- left and right boundaries
			phi2(0,y) = phi2(nxd,y)   
			phi2(nxd+1,y) = phi2(1,y) 
			
		!-- bottom and top boundaries
			phi3(x,0) = phi3(x,nyd)  
			phi3(x,nyd+1) = phi3(x,1)
		!-- left and right boundaries
			phi3(0,y) = phi3(nxd,y)   
			phi3(nxd+1,y) = phi3(1,y) 
		
			

		end if

	end

	attributes(global) subroutine gradient_cal 
	use precision_m
	use device_var, only : nxd, nyd, phi1, phi2, phi3, dphi1dx, dphi2dx, dphi3dx, dphi1dy, dphi2dy, dphi3dy
	implicit none
	
	! TILE must match the thread-block dimensions this kernel is launched with
	! (tblock = dim3(32,32,1) in main.f90). Shared-memory arrays need compile-time
	! constant extents, so this can't just read blockDim%x -- if the launch
	! configuration in main.f90 ever changes, update TILE here to match.
	integer, parameter :: TILE = 32
	
	! One tile per phase, padded with a 1-cell halo ring on every side (indices
	! 0 and TILE+1) so each thread's 8-neighbor stencil can be read entirely from
	! shared memory after the load, with no further global-memory traffic.
	real(fp_kind), shared :: s_phi1(0:TILE+1, 0:TILE+1)
	real(fp_kind), shared :: s_phi2(0:TILE+1, 0:TILE+1)
	real(fp_kind), shared :: s_phi3(0:TILE+1, 0:TILE+1)
	
	integer :: x, y, tx, ty
	x = (blockIdx%x-1)*blockDim%x + threadIdx%x
	y = (blockIdx%y-1)*blockDim%y + threadIdx%y
	tx = threadIdx%x
	ty = threadIdx%y
	
	! Guarded against the *array's* valid bounds (0:nxd+1), not just the interior
	! range this kernel computes over (1:nxd) -- nxd+1 is still a legal index (the
	! periodic halo cell written by periodic_phi), and it must be loadable so the
	! interior thread one column/row in from a block edge can read it as a neighbor.
	! This also correctly skips the kernel's one extra dummy row/column of threads
	! (grid*block = 1024 covers array indices 0:1023, but nxd+1 = 1023 is the last
	! valid index, so thread index 1024 has nothing valid to load).
	if (x<=nxd+1 .and. y<=nyd+1) then
		s_phi1(tx,ty) = phi1(x,y)
		s_phi2(tx,ty) = phi2(x,y)
		s_phi3(tx,ty) = phi3(x,y)
		
		if (tx==1) then
			s_phi1(0,ty) = phi1(x-1,y)
			s_phi2(0,ty) = phi2(x-1,y)
			s_phi3(0,ty) = phi3(x-1,y)
		end if
		if (tx==TILE) then
			s_phi1(TILE+1,ty) = phi1(x+1,y)
			s_phi2(TILE+1,ty) = phi2(x+1,y)
			s_phi3(TILE+1,ty) = phi3(x+1,y)
		end if
		if (ty==1) then
			s_phi1(tx,0) = phi1(x,y-1)
			s_phi2(tx,0) = phi2(x,y-1)
			s_phi3(tx,0) = phi3(x,y-1)
		end if
		if (ty==TILE) then
			s_phi1(tx,TILE+1) = phi1(x,y+1)
			s_phi2(tx,TILE+1) = phi2(x,y+1)
			s_phi3(tx,TILE+1) = phi3(x,y+1)
		end if
		if (tx==1 .and. ty==1) then
			s_phi1(0,0) = phi1(x-1,y-1)
			s_phi2(0,0) = phi2(x-1,y-1)
			s_phi3(0,0) = phi3(x-1,y-1)
		end if
		if (tx==TILE .and. ty==1) then
			s_phi1(TILE+1,0) = phi1(x+1,y-1)
			s_phi2(TILE+1,0) = phi2(x+1,y-1)
			s_phi3(TILE+1,0) = phi3(x+1,y-1)
		end if
		if (tx==1 .and. ty==TILE) then
			s_phi1(0,TILE+1) = phi1(x-1,y+1)
			s_phi2(0,TILE+1) = phi2(x-1,y+1)
			s_phi3(0,TILE+1) = phi3(x-1,y+1)
		end if
		if (tx==TILE .and. ty==TILE) then
			s_phi1(TILE+1,TILE+1) = phi1(x+1,y+1)
			s_phi2(TILE+1,TILE+1) = phi2(x+1,y+1)
			s_phi3(TILE+1,TILE+1) = phi3(x+1,y+1)
		end if
	end if
	
	call syncthreads()
	
	if (x<=nxd .and. x>=1 .and. y>=1 .and.  y<=nyd) then
	
		dphi1dx(x,y) = (s_phi1(tx+1,ty) - s_phi1(tx-1,ty))/3._fp_kind + ( s_phi1(tx+1,ty-1) + s_phi1(tx+1,ty+1) - s_phi1(tx-1,ty-1) - s_phi1(tx-1,ty+1))/12._fp_kind
		dphi1dy(x,y) = (s_phi1(tx,ty+1) - s_phi1(tx,ty-1))/3._fp_kind + ( s_phi1(tx-1,ty+1) + s_phi1(tx+1,ty+1) - s_phi1(tx-1,ty-1) - s_phi1(tx+1,ty-1))/12._fp_kind
		
		dphi2dx(x,y) = (s_phi2(tx+1,ty) - s_phi2(tx-1,ty))/3._fp_kind + ( s_phi2(tx+1,ty-1) + s_phi2(tx+1,ty+1) - s_phi2(tx-1,ty-1) - s_phi2(tx-1,ty+1))/12._fp_kind
		dphi2dy(x,y) = (s_phi2(tx,ty+1) - s_phi2(tx,ty-1))/3._fp_kind + ( s_phi2(tx-1,ty+1) + s_phi2(tx+1,ty+1) - s_phi2(tx-1,ty-1) - s_phi2(tx+1,ty-1))/12._fp_kind
		
		dphi3dx(x,y) = (s_phi3(tx+1,ty) - s_phi3(tx-1,ty))/3._fp_kind + ( s_phi3(tx+1,ty-1) + s_phi3(tx+1,ty+1) - s_phi3(tx-1,ty-1) - s_phi3(tx-1,ty+1))/12._fp_kind
		dphi3dy(x,y) = (s_phi3(tx,ty+1) - s_phi3(tx,ty-1))/3._fp_kind + ( s_phi3(tx-1,ty+1) + s_phi3(tx+1,ty+1) - s_phi3(tx-1,ty-1) - s_phi3(tx+1,ty-1))/12._fp_kind
	
		
	end if
	
	end

	attributes(global) subroutine interface_cal
		use precision_m
		use device_var, only : nxd, nyd, tmp1, tmp2, tmp3, dphi1dx, dphi2dx, dphi3dx, dphi1dy, dphi2dy, dphi3dy, &
		ni1, ni2, ni3, nj1, nj2, nj3
		implicit none
		
		integer :: x, y, n(2)
		x = (blockIdx%x-1)*blockDim%x + threadIdx%x
		y = (blockIdx%y-1)*blockDim%y + threadIdx%y
		n(1) = nxd
		n(2) = nyd!
		
		if (x<=n(1) .and. x>=1 .and. y>=1 .and.  y<=n(2)) then
			
			tmp1(x,y) = sqrt( dphi1dx(x,y)**2._fp_kind + dphi1dy(x,y)**2._fp_kind )
			if(tmp1(x,y)>0) then
				ni1(x,y) = dphi1dx(x,y) / tmp1(x,y)
				nj1(x,y) = dphi1dy(x,y) / tmp1(x,y)

			else
				tmp1(x,y) = sqrt( tmp1(x,y) + 1e-32_fp_kind )
				ni1(x,y) = dphi1dx(x,y) / tmp1(x,y)
				nj1(x,y) = dphi1dy(x,y) / tmp1(x,y)

			end if
				
				
			tmp2(x,y) = sqrt( dphi2dx(x,y)**2._fp_kind + dphi2dy(x,y)**2._fp_kind )
			if(tmp2(x,y)>0) then
				ni2(x,y) = dphi2dx(x,y) / tmp2(x,y)
				nj2(x,y) = dphi2dy(x,y) / tmp2(x,y)

			else
				tmp2(x,y) = sqrt( tmp2(x,y) + 1e-32_fp_kind )
				ni2(x,y) = dphi2dx(x,y) / tmp2(x,y)
				nj2(x,y) = dphi2dy(x,y) / tmp2(x,y)

			end if
				
				
				
			tmp3(x,y) = sqrt( dphi3dx(x,y)**2._fp_kind + dphi3dy(x,y)**2._fp_kind )
			if(tmp3(x,y)>0) then
				ni3(x,y) = dphi3dx(x,y) / tmp3(x,y)
				nj3(x,y) = dphi3dy(x,y) / tmp3(x,y)

			else
				tmp3(x,y) = sqrt( tmp3(x,y) + 1e-32_fp_kind )
				ni3(x,y) = dphi3dx(x,y) / tmp3(x,y)
				nj3(x,y) = dphi3dy(x,y) / tmp3(x,y)

			end if
			
		end if
	end
	
	attributes(global) subroutine chemical_potential
		use precision_m
		use device_var, only : nxd, nyd, lambda, w, lambdat, d2phi1, d2phi2, d2phi3, phi1, phi2, phi3, mu1, mu2, mu3
		implicit none
		
		! Same tiling pattern as gradient_cal: this kernel's Laplacian needs the
		! same 8-neighbor stencil (plus the center point, already needed for mu1/2/3
		! anyway), so it gets the identical shared-memory treatment. See gradient_cal
		! for the detailed reasoning on the load-guard bounds.
		integer, parameter :: TILE = 32
		real(fp_kind), shared :: s_phi1(0:TILE+1, 0:TILE+1)
		real(fp_kind), shared :: s_phi2(0:TILE+1, 0:TILE+1)
		real(fp_kind), shared :: s_phi3(0:TILE+1, 0:TILE+1)
		
		integer :: x, y, tx, ty
		x = (blockIdx%x-1)*blockDim%x + threadIdx%x
		y = (blockIdx%y-1)*blockDim%y + threadIdx%y
		tx = threadIdx%x
		ty = threadIdx%y
		
		if (x<=nxd+1 .and. y<=nyd+1) then
			s_phi1(tx,ty) = phi1(x,y)
			s_phi2(tx,ty) = phi2(x,y)
			s_phi3(tx,ty) = phi3(x,y)
			
			if (tx==1) then
				s_phi1(0,ty) = phi1(x-1,y)
				s_phi2(0,ty) = phi2(x-1,y)
				s_phi3(0,ty) = phi3(x-1,y)
			end if
			if (tx==TILE) then
				s_phi1(TILE+1,ty) = phi1(x+1,y)
				s_phi2(TILE+1,ty) = phi2(x+1,y)
				s_phi3(TILE+1,ty) = phi3(x+1,y)
			end if
			if (ty==1) then
				s_phi1(tx,0) = phi1(x,y-1)
				s_phi2(tx,0) = phi2(x,y-1)
				s_phi3(tx,0) = phi3(x,y-1)
			end if
			if (ty==TILE) then
				s_phi1(tx,TILE+1) = phi1(x,y+1)
				s_phi2(tx,TILE+1) = phi2(x,y+1)
				s_phi3(tx,TILE+1) = phi3(x,y+1)
			end if
			if (tx==1 .and. ty==1) then
				s_phi1(0,0) = phi1(x-1,y-1)
				s_phi2(0,0) = phi2(x-1,y-1)
				s_phi3(0,0) = phi3(x-1,y-1)
			end if
			if (tx==TILE .and. ty==1) then
				s_phi1(TILE+1,0) = phi1(x+1,y-1)
				s_phi2(TILE+1,0) = phi2(x+1,y-1)
				s_phi3(TILE+1,0) = phi3(x+1,y-1)
			end if
			if (tx==1 .and. ty==TILE) then
				s_phi1(0,TILE+1) = phi1(x-1,y+1)
				s_phi2(0,TILE+1) = phi2(x-1,y+1)
				s_phi3(0,TILE+1) = phi3(x-1,y+1)
			end if
			if (tx==TILE .and. ty==TILE) then
				s_phi1(TILE+1,TILE+1) = phi1(x+1,y+1)
				s_phi2(TILE+1,TILE+1) = phi2(x+1,y+1)
				s_phi3(TILE+1,TILE+1) = phi3(x+1,y+1)
			end if
		end if
		
		call syncthreads()
		
		if (x<=nxd .and. x>=1 .and. y>=1 .and.  y<=nyd) then
			
			d2phi1(x,y) = ( s_phi1(tx-1,ty-1)+s_phi1(tx+1,ty-1)+s_phi1(tx-1,ty+1)+s_phi1(tx+1,ty+1) &
				+4._fp_kind*(s_phi1(tx,ty-1)+s_phi1(tx-1,ty)+s_phi1(tx+1,ty)+s_phi1(tx,ty+1)) - 20._fp_kind*s_phi1(tx,ty) )/6._fp_kind
				
			d2phi2(x,y) = ( s_phi2(tx-1,ty-1)+s_phi2(tx+1,ty-1)+s_phi2(tx-1,ty+1)+s_phi2(tx+1,ty+1) &
				+4._fp_kind*(s_phi2(tx,ty-1)+s_phi2(tx-1,ty)+s_phi2(tx+1,ty)+s_phi2(tx,ty+1)) - 20._fp_kind*s_phi2(tx,ty) )/6._fp_kind
				
			d2phi3(x,y) = ( s_phi3(tx-1,ty-1)+s_phi3(tx+1,ty-1)+s_phi3(tx-1,ty+1)+s_phi3(tx+1,ty+1) &
				+4._fp_kind*(s_phi3(tx,ty-1)+s_phi3(tx-1,ty)+s_phi3(tx+1,ty)+s_phi3(tx,ty+1)) - 20._fp_kind*s_phi3(tx,ty) )/6._fp_kind
			
		
			
			mu1(x,y) = (12._fp_kind / w) * (lambda(1) * s_phi1(tx,ty) * (1._fp_kind- s_phi1(tx,ty)) * (1._fp_kind - 2._fp_kind * s_phi1(tx,ty))  &
				- 2._fp_kind * lambdat * s_phi1(tx,ty) * s_phi2(tx,ty) * (1._fp_kind - s_phi1(tx,ty) - s_phi2(tx,ty))) &
					- (3._fp_kind / 4._fp_kind) * w * lambda(1) * d2phi1(x,y) !- 2.0d0 * s_phi1(tx,ty)**2.0d0
					
			mu2(x,y) = (12._fp_kind / w) * (lambda(2) * s_phi2(tx,ty) * (1._fp_kind- s_phi2(tx,ty)) * (1._fp_kind - 2._fp_kind * s_phi2(tx,ty))  &
				- 2._fp_kind * lambdat * s_phi1(tx,ty) * s_phi2(tx,ty) * (1._fp_kind - s_phi1(tx,ty) - s_phi2(tx,ty))) &
					- (3._fp_kind / 4._fp_kind) * w * lambda(2) * d2phi2(x,y) !- 2.0d0 * 0.03d0*s_phi2(tx,ty)**2.0d0
					
			mu3(x,y) = (12._fp_kind / w) * (lambda(3) * s_phi3(tx,ty) * (1._fp_kind- s_phi3(tx,ty)) * (1._fp_kind - 2._fp_kind * s_phi3(tx,ty))  &
				- 2._fp_kind * lambdat * s_phi1(tx,ty) * s_phi2(tx,ty) * (1._fp_kind - s_phi1(tx,ty) - s_phi2(tx,ty))) &
					- (3._fp_kind / 4._fp_kind) * w * lambda(3) * d2phi3(x,y) !- 2.0d0 * 0.03d0*s_phi3(tx,ty)**2.0d0
					
			
						
			
		end if
	
	end
	
	attributes(global) subroutine hydro_cal
		use precision_m
		use device_var, only : nxd, nyd, fpx, fpy, u2, ex, ey, eu0, eu1, eu2, eu3, eu4, eu5, eu6, eu7, eu8, &
		ga_wa0, ga_wa1, ga_wa2, ga_wa3, ga_wa4, ga_wa5, ga_wa6, ga_wa7, ga_wa8, &
		wa, sxx, sxy, syy, fmx, fmy, fsx, fsy, fx, fy, density, tauu, tau1, tau2, tau3, is_solid_node, heav1, heav2, &
		dphi1dx, dphi2dx, dphi3dx, dphi1dy, dphi2dy, dphi3dy, p, ux, uy, mu1, mu2, mu3, rho, phi1, phi2, phi3, &
		g, geq0, geq1, geq2, geq3, geq4, geq5, geq6, geq7, geq8, fg
		
		implicit none
		
		integer :: x, y, n(2)
	
	
		x = (blockIdx%x-1)*blockDim%x + threadIdx%x
		y = (blockIdx%y-1)*blockDim%y + threadIdx%y
		
		n(1) = nxd!size(ux,1)
		n(2) = nyd!size(ux,2)
		
		if (x<=n(1) .and. x>=1 .and. y>=1 .and.  y<=n(2)) then
		!if (x<=n(1) .and. y<=n(2)) then
		
		
			

			p(x,y) = g(x,y,0) + g(x,y,1) + g(x,y,2) + g(x,y,3) + g(x,y,4)+ g(x,y,5) + g(x,y,6) + g(x,y,7) + g(x,y,8)
						
			
		
			
			fpx(x,y) = - p(x,y) * (density(1) * dphi1dx(x,y) + density(2) * dphi2dx(x,y) + density(3) * dphi3dx(x,y)) / 3._fp_kind
			fpy(x,y) = - p(x,y) * (density(1) * dphi1dy(x,y) + density(2) * dphi2dy(x,y) + density(3) * dphi3dy(x,y)) / 3._fp_kind
					

			u2(x,y) = ux(x,y)**2._fp_kind + uy(x,y)**2._fp_kind

			eu0(x,y) = ex(0) * ux(x,y)  + ey(0) * uy(x,y)
			eu1(x,y) = ex(1) * ux(x,y)  + ey(1) * uy(x,y)
			eu2(x,y) = ex(2) * ux(x,y)  + ey(2) * uy(x,y)
			eu3(x,y) = ex(3) * ux(x,y)  + ey(3) * uy(x,y)
			eu4(x,y) = ex(4) * ux(x,y)  + ey(4) * uy(x,y)
			eu5(x,y) = ex(5) * ux(x,y)  + ey(5) * uy(x,y)
			eu6(x,y) = ex(6) * ux(x,y)  + ey(6) * uy(x,y)
			eu7(x,y) = ex(7) * ux(x,y)  + ey(7) * uy(x,y)
			eu8(x,y) = ex(8) * ux(x,y)  + ey(8) * uy(x,y)


			ga_wa0(x,y) = wa(0) * ( eu0(x,y)*(3._fp_kind + 4.5_fp_kind*eu0(x,y)) - 1.5_fp_kind*u2(x,y) )
			ga_wa1(x,y) = wa(1) * ( eu1(x,y)*(3._fp_kind + 4.5_fp_kind*eu1(x,y)) - 1.5_fp_kind*u2(x,y) )
			ga_wa2(x,y) = wa(2) * ( eu2(x,y)*(3._fp_kind + 4.5_fp_kind*eu2(x,y)) - 1.5_fp_kind*u2(x,y) )
			ga_wa3(x,y) = wa(3) * ( eu3(x,y)*(3._fp_kind + 4.5_fp_kind*eu3(x,y)) - 1.5_fp_kind*u2(x,y) )
			ga_wa4(x,y) = wa(4) * ( eu4(x,y)*(3._fp_kind + 4.5_fp_kind*eu4(x,y)) - 1.5_fp_kind*u2(x,y) )
			ga_wa5(x,y) = wa(5) * ( eu5(x,y)*(3._fp_kind + 4.5_fp_kind*eu5(x,y)) - 1.5_fp_kind*u2(x,y) )
			ga_wa6(x,y) = wa(6) * ( eu6(x,y)*(3._fp_kind + 4.5_fp_kind*eu6(x,y)) - 1.5_fp_kind*u2(x,y) )
			ga_wa7(x,y) = wa(7) * ( eu7(x,y)*(3._fp_kind + 4.5_fp_kind*eu7(x,y)) - 1.5_fp_kind*u2(x,y) )
			ga_wa8(x,y) = wa(8) * ( eu8(x,y)*(3._fp_kind + 4.5_fp_kind*eu8(x,y)) - 1.5_fp_kind*u2(x,y) )
				
						
			
			heav1(x,y) = 0.5_fp_kind + sign(0.5_fp_kind,phi1(x,y))
			heav2(x,y) = 0.5_fp_kind + sign(0.5_fp_kind,phi2(x,y))

			tauu(x,y) = (tau1 - tau3) * heav1(x,y) + (tau2 - tau3) * heav2(x,y) + tau3
	
			geq0(x,y) = p(x,y) * wa(0) + ga_wa0(x,y)
			geq1(x,y) = p(x,y) * wa(1) + ga_wa1(x,y)
			geq2(x,y) = p(x,y) * wa(2) + ga_wa2(x,y)
			geq3(x,y) = p(x,y) * wa(3) + ga_wa3(x,y)
			geq4(x,y) = p(x,y) * wa(4) + ga_wa4(x,y)
			geq5(x,y) = p(x,y) * wa(5) + ga_wa5(x,y)
			geq6(x,y) = p(x,y) * wa(6) + ga_wa6(x,y)
			geq7(x,y) = p(x,y) * wa(7) + ga_wa7(x,y)
			geq8(x,y) = p(x,y) * wa(8) + ga_wa8(x,y)
			
			sxx(x,y) = (g(x,y,1) - geq1(x,y)) * ex(1) * ex(1) + &
			(g(x,y,2) - geq2(x,y)) * ex(2) * ex(2) + &
			(g(x,y,3) - geq3(x,y)) * ex(3) * ex(3) + &
			(g(x,y,4) - geq4(x,y)) * ex(4) * ex(4) + &
			(g(x,y,5) - geq5(x,y)) * ex(5) * ex(5) + &
			(g(x,y,6) - geq6(x,y)) * ex(6) * ex(6) + &
			(g(x,y,7) - geq7(x,y)) * ex(7) * ex(7) + &
			(g(x,y,8) - geq8(x,y)) * ex(8) * ex(8)
			
			sxy(x,y) = (g(x,y,1) - geq1(x,y)) * ex(1) * ey(1) + &
			(g(x,y,2) - geq2(x,y)) * ex(2) * ey(2) + &
			(g(x,y,3) - geq3(x,y)) * ex(3) * ey(3) + &
			(g(x,y,4) - geq4(x,y)) * ex(4) * ey(4) + &
			(g(x,y,5) - geq5(x,y)) * ex(5) * ey(5) + &
			(g(x,y,6) - geq6(x,y)) * ex(6) * ey(6) + &
			(g(x,y,7) - geq7(x,y)) * ex(7) * ey(7) + &
			(g(x,y,8) - geq8(x,y)) * ex(8) * ey(8)
			
			syy(x,y) = (g(x,y,1) - geq1(x,y)) * ey(1) * ey(1) + &
			(g(x,y,2) - geq2(x,y)) * ey(2) * ey(2)  + &
			(g(x,y,3) - geq3(x,y)) * ey(3) * ey(3)  + &
			(g(x,y,4) - geq4(x,y)) * ey(4) * ey(4)  + &
			(g(x,y,5) - geq5(x,y)) * ey(5) * ey(5)  + &
			(g(x,y,6) - geq6(x,y)) * ey(6) * ey(6)  + &
			(g(x,y,7) - geq7(x,y)) * ey(7) * ey(7)  + &
			(g(x,y,8) - geq8(x,y)) * ey(8) * ey(8) 
			
			
	
			
			
			
			
			fmx(x,y) = (0.5_fp_kind-tauu(x,y))/tauu(x,y) * (sxx(x,y) * (density(1)*dphi1dx(x,y) + density(2)*dphi2dx(x,y) + density(3)*dphi3dx(x,y)) + &
			sxy(x,y) * (density(1) * dphi1dy(x,y) + density(2) * dphi2dy(x,y) + density(3) * dphi3dy(x,y)))
			
			fmy(x,y) = (0.5_fp_kind-tauu(x,y))/tauu(x,y) * (sxy(x,y) * (density(1)*dphi1dx(x,y) + density(2)*dphi2dx(x,y) + density(3)*dphi3dx(x,y)) + &
			syy(x,y) * (density(1)*dphi1dy(x,y) + density(2)*dphi2dy(x,y) + density(3)*dphi3dy(x,y) ))
	
	
			fsx(x,y) = mu1(x,y)*dphi1dx(x,y) + mu2(x,y)*dphi2dx(x,y) + mu3(x,y)*dphi3dx(x,y)
			fsy(x,y) = mu1(x,y)*dphi1dy(x,y) + mu2(x,y)*dphi2dy(x,y) + mu3(x,y)*dphi3dy(x,y)
						
			!fg(x,y) = 1e-5

			if (is_solid_node(x,y)==0) then
				fx(x,y) = fsx(x,y) + fpx(x,y) + fmx(x,y) !+ fg(x,y)
				fy(x,y) = fsy(x,y) + fpy(x,y) + fmy(x,y) 
			end if
						

			if(is_solid_node(x,y)==0) then
				ux(x,y) = (g(x,y,1)-g(x,y,3)+g(x,y,5)-g(x,y,6)-g(x,y,7)+g(x,y,8)) + 0.5_fp_kind*fx(x,y)/rho(x,y) 
				uy(x,y) = (g(x,y,2)-g(x,y,4)+g(x,y,5)+g(x,y,6)-g(x,y,7)-g(x,y,8)) + 0.5_fp_kind*fy(x,y)/rho(x,y) 
			else
				ux(x,y) = 0
				uy(x,y) = 0
			end if

		end if
	end

	attributes(global) subroutine collision_h
		use precision_m
		use device_var, only : ux, uy, u2, eu0, eu1, eu2, eu3, eu4, eu5, eu6, eu7, eu8, &
		ga_wa0, ga_wa1, ga_wa2, ga_wa3, ga_wa4, ga_wa5, ga_wa6, ga_wa7, ga_wa8, &
		gamma0, gamma1, gamma2, gamma3, gamma4, gamma5, gamma6, gamma7, gamma8, &
		efh10, efh11, efh12, efh13, efh14, efh15, efh16, efh17, efh18, efh20, efh21, efh22, efh23, &
		efh24, efh25, efh26, efh27, efh28, hlph10, hlph11, hlph12, hlph13, hlph14, hlph15, hlph16, &
		hlph17, hlph18, hlph20, hlph21, hlph22, hlph23, hlph24, hlph25, hlph26, hlph27, hlph28, &
		heq10, heq11, heq12, heq13, heq14, heq15, heq16, heq17, heq18, heq20, heq21, heq22, &
		heq23, heq24, heq25, heq26, heq27, heq28, phi1, phi2, phi3, ni1, ni2, ni3, nj1, nj2, nj3, &
		h1, h2, is_solid_node, ex, ey, wa, w, w_c
		implicit none
		!real(fp_kind), shared, dimension(32,32) :: h10_s, h11_s, h12_s, h13_s, h14_s, h15_s, h16_s, h17_s, h18_s
		!real(fp_kind), shared, dimension(16,16) :: h20_s, h21_s, h22_s, h23_s, h24_s, h25_s, h26_s, h27_s, h28_s
		integer :: x, y, n(2), xs, ys
		
		x = (blockIdx%x-1)*blockDim%x + threadIdx%x
		y = (blockIdx%y-1)*blockDim%y + threadIdx%y
		xs = threadIdx%x
		ys = threadIdx%y
		n(1) = size(ux,1)
		n(2) = size(ux,2)
	
	
		if (x<=n(1) .and. x>=1 .and. y>=1 .and.  y<=n(2)) then

			! h10_s(xs,ys) = h1(x,y,0)
			! h11_s(xs,ys) = h1(x,y,1)
			! h12_s(xs,ys) = h1(x,y,2)
			! h13_s(xs,ys) = h1(x,y,3)
			! h14_s(xs,ys) = h1(x,y,4)
			! h15_s(xs,ys) = h1(x,y,5)
			! h16_s(xs,ys) = h1(x,y,6)
			! h17_s(xs,ys) = h1(x,y,7)
			! h18_s(xs,ys) = h1(x,y,8)
			
			! h20_s(xs,ys) = h2(x,y,0)
			! h21_s(xs,ys) = h2(x,y,1)
			! h22_s(xs,ys) = h2(x,y,2)
			! h23_s(xs,ys) = h2(x,y,3)
			! h24_s(xs,ys) = h2(x,y,4)
			! h25_s(xs,ys) = h2(x,y,5)
			! h26_s(xs,ys) = h2(x,y,6)
			! h27_s(xs,ys) = h2(x,y,7)
			! h28_s(xs,ys) = h2(x,y,8)
				
			u2(x,y) = ux(x,y)**2._fp_kind + uy(x,y)**2._fp_kind
			
			eu0(x,y) = ex(0) * ux(x,y)  + ey(0) * uy(x,y)
			eu1(x,y) = ex(1) * ux(x,y)  + ey(1) * uy(x,y)
			eu2(x,y) = ex(2) * ux(x,y)  + ey(2) * uy(x,y)
			eu3(x,y) = ex(3) * ux(x,y)  + ey(3) * uy(x,y)
			eu4(x,y) = ex(4) * ux(x,y)  + ey(4) * uy(x,y)
			eu5(x,y) = ex(5) * ux(x,y)  + ey(5) * uy(x,y)
			eu6(x,y) = ex(6) * ux(x,y)  + ey(6) * uy(x,y)
			eu7(x,y) = ex(7) * ux(x,y)  + ey(7) * uy(x,y)
			eu8(x,y) = ex(8) * ux(x,y)  + ey(8) * uy(x,y)
			
			ga_wa0(x,y) = wa(0) * ( eu0(x,y)*(3._fp_kind + 4.5_fp_kind*eu0(x,y)) - 1.5_fp_kind*u2(x,y) )
			ga_wa1(x,y) = wa(1) * ( eu1(x,y)*(3._fp_kind + 4.5_fp_kind*eu1(x,y)) - 1.5_fp_kind*u2(x,y) )
			ga_wa2(x,y) = wa(2) * ( eu2(x,y)*(3._fp_kind + 4.5_fp_kind*eu2(x,y)) - 1.5_fp_kind*u2(x,y) )
			ga_wa3(x,y) = wa(3) * ( eu3(x,y)*(3._fp_kind + 4.5_fp_kind*eu3(x,y)) - 1.5_fp_kind*u2(x,y) )
			ga_wa4(x,y) = wa(4) * ( eu4(x,y)*(3._fp_kind + 4.5_fp_kind*eu4(x,y)) - 1.5_fp_kind*u2(x,y) )
			ga_wa5(x,y) = wa(5) * ( eu5(x,y)*(3._fp_kind + 4.5_fp_kind*eu5(x,y)) - 1.5_fp_kind*u2(x,y) )
			ga_wa6(x,y) = wa(6) * ( eu6(x,y)*(3._fp_kind + 4.5_fp_kind*eu6(x,y)) - 1.5_fp_kind*u2(x,y) )
			ga_wa7(x,y) = wa(7) * ( eu7(x,y)*(3._fp_kind + 4.5_fp_kind*eu7(x,y)) - 1.5_fp_kind*u2(x,y) )
			ga_wa8(x,y) = wa(8) * ( eu8(x,y)*(3._fp_kind + 4.5_fp_kind*eu8(x,y)) - 1.5_fp_kind*u2(x,y) )
		

			gamma0(x,y) = ga_wa0(x,y) + wa(0)
			gamma1(x,y) = ga_wa1(x,y) + wa(1)
			gamma2(x,y) = ga_wa2(x,y) + wa(2)
			gamma3(x,y) = ga_wa3(x,y) + wa(3)
			gamma4(x,y) = ga_wa4(x,y) + wa(4)
			gamma5(x,y) = ga_wa5(x,y) + wa(5)
			gamma6(x,y) = ga_wa6(x,y) + wa(6)
			gamma7(x,y) = ga_wa7(x,y) + wa(7)
			gamma8(x,y) = ga_wa8(x,y) + wa(8)
			
			
			
			efh10(x,y)  = 2._fp_kind * ((4._fp_kind/w) * abs(phi1(x,y) * (1._fp_kind - phi1(x,y))) * (ex(0) * ni1(x,y) + ey(0) * nj1(x,y))) &
				- ((4._fp_kind/w) * abs(phi2(x,y) * (1._fp_kind - phi2(x,y))) * (ex(0) * ni2(x,y) + ey(0) * nj2(x,y)) &
				+ (4._fp_kind/w) * abs(phi3(x,y) * (1._fp_kind - phi3(x,y))) * (ex(0) * ni3(x,y) + ey(0) * nj3(x,y)))
				
			efh11(x,y)  = 2._fp_kind * ((4._fp_kind/w) * abs(phi1(x,y) * (1._fp_kind - phi1(x,y))) * (ex(1) * ni1(x,y) + ey(1) * nj1(x,y))) &
				- ((4._fp_kind/w) * abs(phi2(x,y) * (1._fp_kind - phi2(x,y))) * (ex(1) * ni2(x,y) + ey(1) * nj2(x,y)) &
				+ (4._fp_kind/w) * abs(phi3(x,y) * (1._fp_kind - phi3(x,y))) * (ex(1) * ni3(x,y) + ey(1) * nj3(x,y)))
				
			efh12(x,y)  = 2._fp_kind * ((4._fp_kind/w) * abs(phi1(x,y) * (1._fp_kind - phi1(x,y))) * (ex(2) * ni1(x,y) + ey(2) * nj1(x,y))) &
				- ((4._fp_kind/w) * abs(phi2(x,y) * (1._fp_kind - phi2(x,y))) * (ex(2) * ni2(x,y) + ey(2) * nj2(x,y)) &
				+ (4._fp_kind/w) * abs(phi3(x,y) * (1._fp_kind - phi3(x,y))) * (ex(2) * ni3(x,y) + ey(2) * nj3(x,y)))
				
			efh13(x,y)  = 2._fp_kind * ((4._fp_kind/w) * abs(phi1(x,y) * (1._fp_kind - phi1(x,y))) * (ex(3) * ni1(x,y) + ey(3) * nj1(x,y))) &
				- ((4._fp_kind/w) * abs(phi2(x,y) * (1._fp_kind - phi2(x,y))) * (ex(3) * ni2(x,y) + ey(3) * nj2(x,y)) &
				+ (4._fp_kind/w) * abs(phi3(x,y) * (1._fp_kind - phi3(x,y))) * (ex(3) * ni3(x,y) + ey(3) * nj3(x,y)))
				
			efh14(x,y)  = 2._fp_kind * ((4._fp_kind/w) * abs(phi1(x,y) * (1._fp_kind - phi1(x,y))) * (ex(4) * ni1(x,y) + ey(4) * nj1(x,y))) &
				- ((4._fp_kind/w) * abs(phi2(x,y) * (1._fp_kind - phi2(x,y))) * (ex(4) * ni2(x,y) + ey(4) * nj2(x,y)) &
				+ (4._fp_kind/w) * abs(phi3(x,y) * (1._fp_kind - phi3(x,y))) * (ex(4) * ni3(x,y) + ey(4) * nj3(x,y)))
				
			efh15(x,y)  = 2._fp_kind * ((4._fp_kind/w) * abs(phi1(x,y) * (1._fp_kind - phi1(x,y))) * (ex(5) * ni1(x,y) + ey(5) * nj1(x,y))) &
				- ((4._fp_kind/w) * abs(phi2(x,y) * (1._fp_kind - phi2(x,y))) * (ex(5) * ni2(x,y) + ey(5) * nj2(x,y)) &
				+ (4._fp_kind/w) * abs(phi3(x,y) * (1._fp_kind - phi3(x,y))) * (ex(5) * ni3(x,y) + ey(5) * nj3(x,y)))
				
			efh16(x,y)  = 2._fp_kind * ((4._fp_kind/w) * abs(phi1(x,y) * (1._fp_kind - phi1(x,y))) * (ex(6) * ni1(x,y) + ey(6) * nj1(x,y))) &
				- ((4._fp_kind/w) * abs(phi2(x,y) * (1._fp_kind - phi2(x,y))) * (ex(6) * ni2(x,y) + ey(6) * nj2(x,y)) &
				+ (4._fp_kind/w) * abs(phi3(x,y) * (1._fp_kind - phi3(x,y))) * (ex(6) * ni3(x,y) + ey(6) * nj3(x,y)))
				
			efh17(x,y)  = 2._fp_kind * ((4._fp_kind/w) * abs(phi1(x,y) * (1._fp_kind - phi1(x,y))) * (ex(7) * ni1(x,y) + ey(7) * nj1(x,y))) &
				- ((4._fp_kind/w) * abs(phi2(x,y) * (1._fp_kind - phi2(x,y))) * (ex(7) * ni2(x,y) + ey(7) * nj2(x,y)) &
				+ (4._fp_kind/w) * abs(phi3(x,y) * (1._fp_kind - phi3(x,y))) * (ex(7) * ni3(x,y) + ey(7) * nj3(x,y)))
				
			efh18(x,y)  = 2._fp_kind * ((4._fp_kind/w) * abs(phi1(x,y) * (1._fp_kind - phi1(x,y))) * (ex(8) * ni1(x,y) + ey(8) * nj1(x,y))) &
				- ((4._fp_kind/w) * abs(phi2(x,y) * (1._fp_kind - phi2(x,y))) * (ex(8) * ni2(x,y) + ey(8) * nj2(x,y)) &
				+ (4._fp_kind/w) * abs(phi3(x,y) * (1._fp_kind - phi3(x,y))) * (ex(8) * ni3(x,y) + ey(8) * nj3(x,y)))
	
			 
			 

			efh20(x,y)  = 2._fp_kind * ((4._fp_kind/w) * abs(phi2(x,y) * (1._fp_kind - phi2(x,y))) * (ex(0) * ni2(x,y) + ey(0) * nj2(x,y))) &
				- ((4._fp_kind/w) * abs(phi1(x,y) * (1._fp_kind - phi1(x,y))) * (ex(0) * ni1(x,y) + ey(0) * nj1(x,y)) &
				+ (4._fp_kind/w) * abs(phi3(x,y) * (1._fp_kind - phi3(x,y))) * (ex(0) * ni3(x,y) + ey(0) * nj3(x,y)))
				
			efh21(x,y)  = 2._fp_kind * ((4._fp_kind/w) * abs(phi2(x,y) * (1._fp_kind - phi2(x,y))) * (ex(1) * ni2(x,y) + ey(1) * nj2(x,y))) &
				- ((4._fp_kind/w) * abs(phi1(x,y) * (1._fp_kind - phi1(x,y))) * (ex(1) * ni1(x,y) + ey(1) * nj1(x,y)) &
				+ (4._fp_kind/w) * abs(phi3(x,y) * (1._fp_kind - phi3(x,y))) * (ex(1) * ni3(x,y) + ey(1) * nj3(x,y)))
				
			efh22(x,y)  = 2._fp_kind * ((4._fp_kind/w) * abs(phi2(x,y) * (1._fp_kind - phi2(x,y))) * (ex(2) * ni2(x,y) + ey(2) * nj2(x,y))) &
				- ((4._fp_kind/w) * abs(phi1(x,y) * (1._fp_kind - phi1(x,y))) * (ex(2) * ni1(x,y) + ey(2) * nj1(x,y)) &
				+ (4._fp_kind/w) * abs(phi3(x,y) * (1._fp_kind - phi3(x,y))) * (ex(2) * ni3(x,y) + ey(2) * nj3(x,y)))
				
			efh23(x,y)  = 2._fp_kind * ((4._fp_kind/w) * abs(phi2(x,y) * (1._fp_kind - phi2(x,y))) * (ex(3) * ni2(x,y) + ey(3) * nj2(x,y))) &
				- ((4._fp_kind/w) * abs(phi1(x,y) * (1._fp_kind - phi1(x,y))) * (ex(3) * ni1(x,y) + ey(3) * nj1(x,y)) &
				+ (4._fp_kind/w) * abs(phi3(x,y) * (1._fp_kind - phi3(x,y))) * (ex(3) * ni3(x,y) + ey(3) * nj3(x,y)))
				
			efh24(x,y)  = 2._fp_kind * ((4._fp_kind/w) * abs(phi2(x,y) * (1._fp_kind - phi2(x,y))) * (ex(4) * ni2(x,y) + ey(4) * nj2(x,y))) &
				- ((4._fp_kind/w) * abs(phi1(x,y) * (1._fp_kind - phi1(x,y))) * (ex(4) * ni1(x,y) + ey(4) * nj1(x,y)) &
				+ (4._fp_kind/w) * abs(phi3(x,y) * (1._fp_kind - phi3(x,y))) * (ex(4) * ni3(x,y) + ey(4) * nj3(x,y)))
				
			efh25(x,y)  = 2._fp_kind * ((4._fp_kind/w) * abs(phi2(x,y) * (1._fp_kind - phi2(x,y))) * (ex(5) * ni2(x,y) + ey(5) * nj2(x,y))) &
				- ((4._fp_kind/w) * abs(phi1(x,y) * (1._fp_kind - phi1(x,y))) * (ex(5) * ni1(x,y) + ey(5) * nj1(x,y)) &
				+ (4._fp_kind/w) * abs(phi3(x,y) * (1._fp_kind - phi3(x,y))) * (ex(5) * ni3(x,y) + ey(5) * nj3(x,y)))
				
			efh26(x,y)  = 2._fp_kind * ((4._fp_kind/w) * abs(phi2(x,y) * (1._fp_kind - phi2(x,y))) * (ex(6) * ni2(x,y) + ey(6) * nj2(x,y))) &
				- ((4._fp_kind/w) * abs(phi1(x,y) * (1._fp_kind - phi1(x,y))) * (ex(6) * ni1(x,y) + ey(6) * nj1(x,y)) &
				+ (4._fp_kind/w) * abs(phi3(x,y) * (1._fp_kind - phi3(x,y))) * (ex(6) * ni3(x,y) + ey(6) * nj3(x,y)))
				
			efh27(x,y)  = 2._fp_kind * ((4._fp_kind/w) * abs(phi2(x,y) * (1._fp_kind - phi2(x,y))) * (ex(7) * ni2(x,y) + ey(7) * nj2(x,y))) &
				- ((4._fp_kind/w) * abs(phi1(x,y) * (1._fp_kind - phi1(x,y))) * (ex(7) * ni1(x,y) + ey(7) * nj1(x,y)) &
				+ (4._fp_kind/w) * abs(phi3(x,y) * (1._fp_kind - phi3(x,y))) * (ex(7) * ni3(x,y) + ey(7) * nj3(x,y)))
				
			efh28(x,y)  = 2._fp_kind * ((4._fp_kind/w) * abs(phi2(x,y) * (1._fp_kind - phi2(x,y))) * (ex(8) * ni2(x,y) + ey(8) * nj2(x,y))) &
				- ((4._fp_kind/w) * abs(phi1(x,y) * (1._fp_kind - phi1(x,y))) * (ex(8) * ni1(x,y) + ey(8) * nj1(x,y)) &
				+ (4._fp_kind/w) * abs(phi3(x,y) * (1._fp_kind - phi3(x,y))) * (ex(8) * ni3(x,y) + ey(8) * nj3(x,y)))
	


			
			hlph10(x,y) = wa(0) * efh10(x,y) / 3._fp_kind
			hlph11(x,y) = wa(1) * efh11(x,y) / 3._fp_kind
			hlph12(x,y) = wa(2) * efh12(x,y) / 3._fp_kind
			hlph13(x,y) = wa(3) * efh13(x,y) / 3._fp_kind
			hlph14(x,y) = wa(4) * efh14(x,y) / 3._fp_kind
			hlph15(x,y) = wa(5) * efh15(x,y) / 3._fp_kind
			hlph16(x,y) = wa(6) * efh16(x,y) / 3._fp_kind
			hlph17(x,y) = wa(7) * efh17(x,y) / 3._fp_kind
			hlph18(x,y) = wa(8) * efh18(x,y) / 3._fp_kind
			
			hlph20(x,y) = wa(0) * efh20(x,y) / 3._fp_kind
			hlph21(x,y) = wa(1) * efh21(x,y) / 3._fp_kind
			hlph22(x,y) = wa(2) * efh22(x,y) / 3._fp_kind
			hlph23(x,y) = wa(3) * efh23(x,y) / 3._fp_kind
			hlph24(x,y) = wa(4) * efh24(x,y) / 3._fp_kind
			hlph25(x,y) = wa(5) * efh25(x,y) / 3._fp_kind
			hlph26(x,y) = wa(6) * efh26(x,y) / 3._fp_kind
			hlph27(x,y) = wa(7) * efh27(x,y) / 3._fp_kind
			hlph28(x,y) = wa(8) * efh28(x,y) / 3._fp_kind
			
			

			heq10(x,y) = phi1(x,y) * gamma0(x,y) - 0.5_fp_kind * hlph10(x,y)
			heq11(x,y) = phi1(x,y) * gamma1(x,y) - 0.5_fp_kind * hlph11(x,y)
			heq12(x,y) = phi1(x,y) * gamma2(x,y) - 0.5_fp_kind * hlph12(x,y)
			heq13(x,y) = phi1(x,y) * gamma3(x,y) - 0.5_fp_kind * hlph13(x,y)
			heq14(x,y) = phi1(x,y) * gamma4(x,y) - 0.5_fp_kind * hlph14(x,y)
			heq15(x,y) = phi1(x,y) * gamma5(x,y) - 0.5_fp_kind * hlph15(x,y)
			heq16(x,y) = phi1(x,y) * gamma6(x,y) - 0.5_fp_kind * hlph16(x,y)
			heq17(x,y) = phi1(x,y) * gamma7(x,y) - 0.5_fp_kind * hlph17(x,y)
			heq18(x,y) = phi1(x,y) * gamma8(x,y) - 0.5_fp_kind * hlph18(x,y)
			
			
			heq20(x,y) = phi2(x,y) * gamma0(x,y) - 0.5_fp_kind * hlph20(x,y)
			heq21(x,y) = phi2(x,y) * gamma1(x,y) - 0.5_fp_kind * hlph21(x,y)
			heq22(x,y) = phi2(x,y) * gamma2(x,y) - 0.5_fp_kind * hlph22(x,y)
			heq23(x,y) = phi2(x,y) * gamma3(x,y) - 0.5_fp_kind * hlph23(x,y)
			heq24(x,y) = phi2(x,y) * gamma4(x,y) - 0.5_fp_kind * hlph24(x,y)
			heq25(x,y) = phi2(x,y) * gamma5(x,y) - 0.5_fp_kind * hlph25(x,y)
			heq26(x,y) = phi2(x,y) * gamma6(x,y) - 0.5_fp_kind * hlph26(x,y)
			heq27(x,y) = phi2(x,y) * gamma7(x,y) - 0.5_fp_kind * hlph27(x,y)
			heq28(x,y) = phi2(x,y) * gamma8(x,y) - 0.5_fp_kind * hlph28(x,y)
	
			if (is_solid_node(x,y)==0) then
			
				h1(x,y,0) = h1(x,y,0) * (1._fp_kind-w_c) + heq10(x,y) * w_c +  hlph10(x,y)
				h1(x,y,1) = h1(x,y,1) * (1._fp_kind-w_c) + heq11(x,y) * w_c +  hlph11(x,y)
				h1(x,y,2) = h1(x,y,2) * (1._fp_kind-w_c) + heq12(x,y) * w_c +  hlph12(x,y)
				h1(x,y,3) = h1(x,y,3) * (1._fp_kind-w_c) + heq13(x,y) * w_c +  hlph13(x,y)
				h1(x,y,4) = h1(x,y,4) * (1._fp_kind-w_c) + heq14(x,y) * w_c +  hlph14(x,y)
				h1(x,y,5) = h1(x,y,5) * (1._fp_kind-w_c) + heq15(x,y) * w_c +  hlph15(x,y)
				h1(x,y,6) = h1(x,y,6) * (1._fp_kind-w_c) + heq16(x,y) * w_c +  hlph16(x,y)
				h1(x,y,7) = h1(x,y,7) * (1._fp_kind-w_c) + heq17(x,y) * w_c +  hlph17(x,y)
				h1(x,y,8) = h1(x,y,8) * (1._fp_kind-w_c) + heq18(x,y) * w_c +  hlph18(x,y)
				
				! h10_s(xs,ys) = h10_s(xs,ys) * (1.-w_c) + heq10(x,y) * w_c +  hlph10(x,y)
				! h11_s(xs,ys) = h11_s(xs,ys) * (1.-w_c) + heq11(x,y) * w_c +  hlph11(x,y)
				! h12_s(xs,ys) = h12_s(xs,ys) * (1.-w_c) + heq12(x,y) * w_c +  hlph12(x,y)
				! h13_s(xs,ys) = h13_s(xs,ys) * (1.-w_c) + heq13(x,y) * w_c +  hlph13(x,y)
				! h14_s(xs,ys) = h14_s(xs,ys) * (1.-w_c) + heq14(x,y) * w_c +  hlph14(x,y)
				! h15_s(xs,ys) = h15_s(xs,ys) * (1.-w_c) + heq15(x,y) * w_c +  hlph15(x,y)
				! h16_s(xs,ys) = h16_s(xs,ys) * (1.-w_c) + heq16(x,y) * w_c +  hlph16(x,y)
				! h17_s(xs,ys) = h17_s(xs,ys) * (1.-w_c) + heq17(x,y) * w_c +  hlph17(x,y)
				! h18_s(xs,ys) = h18_s(xs,ys) * (1.-w_c) + heq18(x,y) * w_c +  hlph18(x,y)
				
				
				
				
				h2(x,y,0) = h2(x,y,0) * (1._fp_kind-w_c) + heq20(x,y) * w_c +  hlph20(x,y)
				h2(x,y,1) = h2(x,y,1) * (1._fp_kind-w_c) + heq21(x,y) * w_c +  hlph21(x,y)
				h2(x,y,2) = h2(x,y,2) * (1._fp_kind-w_c) + heq22(x,y) * w_c +  hlph22(x,y)
				h2(x,y,3) = h2(x,y,3) * (1._fp_kind-w_c) + heq23(x,y) * w_c +  hlph23(x,y)
				h2(x,y,4) = h2(x,y,4) * (1._fp_kind-w_c) + heq24(x,y) * w_c +  hlph24(x,y)
				h2(x,y,5) = h2(x,y,5) * (1._fp_kind-w_c) + heq25(x,y) * w_c +  hlph25(x,y)
				h2(x,y,6) = h2(x,y,6) * (1._fp_kind-w_c) + heq26(x,y) * w_c +  hlph26(x,y)
				h2(x,y,7) = h2(x,y,7) * (1._fp_kind-w_c) + heq27(x,y) * w_c +  hlph27(x,y)
				h2(x,y,8) = h2(x,y,8) * (1._fp_kind-w_c) + heq28(x,y) * w_c +  hlph28(x,y)
				
				! h20_s(xs,ys) = h20_s(xs,ys) * (1.-w_c) + heq20(x,y) * w_c +  hlph20(x,y)
				! h21_s(xs,ys) = h21_s(xs,ys) * (1.-w_c) + heq21(x,y) * w_c +  hlph21(x,y)
				! h22_s(xs,ys) = h22_s(xs,ys) * (1.-w_c) + heq22(x,y) * w_c +  hlph22(x,y)
				! h23_s(xs,ys) = h23_s(xs,ys) * (1.-w_c) + heq23(x,y) * w_c +  hlph23(x,y)
				! h24_s(xs,ys) = h24_s(xs,ys) * (1.-w_c) + heq24(x,y) * w_c +  hlph24(x,y)
				! h25_s(xs,ys) = h25_s(xs,ys) * (1.-w_c) + heq25(x,y) * w_c +  hlph25(x,y)
				! h26_s(xs,ys) = h26_s(xs,ys) * (1.-w_c) + heq26(x,y) * w_c +  hlph26(x,y)
				! h27_s(xs,ys) = h27_s(xs,ys) * (1.-w_c) + heq27(x,y) * w_c +  hlph27(x,y)
				! h28_s(xs,ys) = h28_s(xs,ys) * (1.-w_c) + heq28(x,y) * w_c +  hlph28(x,y)
				
				!call syncthreads()
				
				! h1(x,y,0) = h10_s(xs, ys)
				! h1(x,y,1) = h11_s(xs, ys)
				! h1(x,y,2) = h12_s(xs, ys)
				! h1(x,y,3) = h13_s(xs, ys)
				! h1(x,y,4) = h14_s(xs, ys)
				! h1(x,y,5) = h15_s(xs, ys)
				! h1(x,y,6) = h16_s(xs, ys)
				! h1(x,y,7) = h17_s(xs, ys)
				! h1(x,y,8) = h18_s(xs, ys)
				
				! h2(x,y,0) = h20_s(xs, ys)
				! h2(x,y,1) = h21_s(xs, ys)
				! h2(x,y,2) = h22_s(xs, ys)
				! h2(x,y,3) = h23_s(xs, ys)
				! h2(x,y,4) = h24_s(xs, ys)
				! h2(x,y,5) = h25_s(xs, ys)
				! h2(x,y,6) = h26_s(xs, ys)
				! h2(x,y,7) = h27_s(xs, ys)
				! h2(x,y,8) = h28_s(xs, ys)
			
				
				

			end if
					
					
					
					
					
					
				
					
		end if
		
		
							
	end
	
	attributes(global) subroutine collision_g1
		use precision_m
		use device_var, only : nxd, nyd, u2, ex, ey, ga_wa0, ga_wa1, ga_wa2, ga_wa3, ga_wa4, ga_wa5, ga_wa6, &
		ga_wa7, ga_wa8, wa, fpx, fpy, &
		tauu, sxx, sxy, syy, fmx, fmy, fsx, fsy, fx, fy, density, is_solid_node, phi1, phi2, phi3, &
		ux, uy, mu1, mu2, mu3, dphi1dx, dphi2dx, dphi3dx, dphi1dy, dphi2dy, dphi3dy, p, rho, &
		geq0, geq1, geq2, geq3, geq4, geq5, geq6, geq7, geq8, g, &
		hlpg0, hlpg1, hlpg2, hlpg3, hlpg4, hlpg5, hlpg6, hlpg7, hlpg8, ef0, ef1, ef2, ef3, ef4, ef5, ef6, ef7, ef8
		implicit none
		
		integer :: x, y, n(2)
		
		x = (blockIdx%x-1)*blockDim%x + threadIdx%x
		y = (blockIdx%y-1)*blockDim%y + threadIdx%y
		n(1) = size(ux,1)
		n(2) = size(ux,2)
	
	
		!if (x<=n(1) .and. y<=n(2)) then
		if (x<=n(1) .and. x>=1 .and. y>=1 .and.  y<=n(2)) then
						

			
			
			
			geq0(x,y) = p(x,y) * wa(0) + ga_wa0(x,y)
			geq1(x,y) = p(x,y) * wa(1) + ga_wa1(x,y)
			geq2(x,y) = p(x,y) * wa(2) + ga_wa2(x,y)
			geq3(x,y) = p(x,y) * wa(3) + ga_wa3(x,y)
			geq4(x,y) = p(x,y) * wa(4) + ga_wa4(x,y)
			geq5(x,y) = p(x,y) * wa(5) + ga_wa5(x,y)
			geq6(x,y) = p(x,y) * wa(6) + ga_wa6(x,y)
			geq7(x,y) = p(x,y) * wa(7) + ga_wa7(x,y)
			geq8(x,y) = p(x,y) * wa(8) + ga_wa8(x,y)
			
			
			sxx(x,y) = (g(x,y,1) - geq1(x,y)) * ex(1) * ex(1) + &
			(g(x,y,2) - geq2(x,y)) * ex(2) * ex(2) + &
			(g(x,y,3) - geq3(x,y)) * ex(3) * ex(3) + &
			(g(x,y,4) - geq4(x,y)) * ex(4) * ex(4) + &
			(g(x,y,5) - geq5(x,y)) * ex(5) * ex(5) + &
			(g(x,y,6) - geq6(x,y)) * ex(6) * ex(6) + &
			(g(x,y,7) - geq7(x,y)) * ex(7) * ex(7) + &
			(g(x,y,8) - geq8(x,y)) * ex(8) * ex(8)

			sxy(x,y) = (g(x,y,1) - geq1(x,y)) * ex(1) * ey(1) + &
			(g(x,y,2) - geq2(x,y)) * ex(2) * ey(2) + &
			(g(x,y,3) - geq3(x,y)) * ex(3) * ey(3) + &
			(g(x,y,4) - geq4(x,y)) * ex(4) * ey(4) + &
			(g(x,y,5) - geq5(x,y)) * ex(5) * ey(5) + &
			(g(x,y,6) - geq6(x,y)) * ex(6) * ey(6) + &
			(g(x,y,7) - geq7(x,y)) * ex(7) * ey(7) + &
			(g(x,y,8) - geq8(x,y)) * ex(8) * ey(8)

			syy(x,y) = (g(x,y,1) - geq1(x,y)) * ey(1) * ey(1) + &
			(g(x,y,2) - geq2(x,y)) * ey(2) * ey(2)  + &
			(g(x,y,3) - geq3(x,y)) * ey(3) * ey(3)  + &
			(g(x,y,4) - geq4(x,y)) * ey(4) * ey(4)  + &
			(g(x,y,5) - geq5(x,y)) * ey(5) * ey(5)  + &
			(g(x,y,6) - geq6(x,y)) * ey(6) * ey(6)  + &
			(g(x,y,7) - geq7(x,y)) * ey(7) * ey(7)  + &
			(g(x,y,8) - geq8(x,y)) * ey(8) * ey(8) 
			
			
			
			fmx(x,y) = (0.5_fp_kind-tauu(x,y))/tauu(x,y) * (sxx(x,y) * (density(1)*dphi1dx(x,y) + density(2)*dphi2dx(x,y) + density(3)*dphi3dx(x,y)) + &
			sxy(x,y) * (density(1) * dphi1dy(x,y) + density(2) * dphi2dy(x,y) + density(3) * dphi3dy(x,y)))

			fmy(x,y) = (0.5_fp_kind-tauu(x,y))/tauu(x,y) * (sxy(x,y) * (density(1)*dphi1dx(x,y) + density(2)*dphi2dx(x,y) + density(3)*dphi3dx(x,y)) + &
			syy(x,y) * (density(1)*dphi1dy(x,y) + density(2)*dphi2dy(x,y) + density(3)*dphi3dy(x,y) ))


					

				
		end if
		
		
							
	end
	
	attributes(global) subroutine collision_g2
		use precision_m
		use device_var, only : nxd, nyd, ex, ey, &
		ga_wa0, ga_wa1, ga_wa2, ga_wa3, ga_wa4, ga_wa5, ga_wa6, ga_wa7, ga_wa8, &
		wa, fx, fy, fsx, fsy, fpx, fpy, fmx, fmy, tauu, is_solid_node, ef0, ef1, ef2, ef3, ef4, ef5, ef6, ef7, ef8, &
		p, ux, uy, rho, g, geq0, geq1, geq2, geq3, geq4, geq5, geq6, geq7, geq8, &
		hlpg0, hlpg1, hlpg2, hlpg3, hlpg4, hlpg5, hlpg6, hlpg7, hlpg8, fg
		implicit none
		! real(fp_kind), intent(inout) :: h(2,0:8,0:nxd+1,0:nyd+1), g(0:8,0:nxd+1,0:nyd+1)
		! real(fp_kind), intent(in) :: phi(3,0:nxd+1,0:nyd+1), ni(3,nxd,nyd), nj(3,nxd,nyd), mu(3,nxd,nyd), dphidx(3,nxd,nyd), dphidy(3,nxd,nyd)
		! real(fp_kind), intent(in) :: p(nxd,nyd), rho(nxd,nyd), ux(nxd, nyd), uy(nxd,nyd)
		integer :: x, y, n(2)
		
		!real(fp_kind) :: fsx(nxd,nyd), fsy(nxd,nyd), fpx(nxd,nyd), fpy(nxd,nyd), ef(0:8,nxd,nyd), hlpg(0:8,nxd,nyd), geq(0:8,nxd,nyd), fx(nxd,nyd), fy(nxd,nyd)
		x = (blockIdx%x-1)*blockDim%x + threadIdx%x
		y = (blockIdx%y-1)*blockDim%y + threadIdx%y
		n(1) = nxd!size(ux,1)
		n(2) = nyd!size(ux,2)
	
	
		if (x<=n(1) .and. x>=1 .and. y>=1 .and.  y<=n(2)) then
						
						
						


			if (is_solid_node(x,y)==0) then
				fx(x,y) = fsx(x,y) + fpx(x,y) + fmx(x,y) 
				fy(x,y) = fsy(x,y) + fpy(x,y) + fmy(x,y) 
			end if
						
				

			ef0(x,y) = ex(0) * fx(x,y) + ey(0) * fy(x,y)
			ef1(x,y) = ex(1) * fx(x,y) + ey(1) * fy(x,y)
			ef2(x,y) = ex(2) * fx(x,y) + ey(2) * fy(x,y)
			ef3(x,y) = ex(3) * fx(x,y) + ey(3) * fy(x,y)
			ef4(x,y) = ex(4) * fx(x,y) + ey(4) * fy(x,y)
			ef5(x,y) = ex(5) * fx(x,y) + ey(5) * fy(x,y)
			ef6(x,y) = ex(6) * fx(x,y) + ey(6) * fy(x,y)
			ef7(x,y) = ex(7) * fx(x,y) + ey(7) * fy(x,y)
			ef8(x,y) = ex(8) * fx(x,y) + ey(8) * fy(x,y)
			

			hlpg0(x,y) = 3._fp_kind * wa(0) * ef0(x,y) / rho(x,y)
			hlpg1(x,y) = 3._fp_kind * wa(1) * ef1(x,y) / rho(x,y)
			hlpg2(x,y) = 3._fp_kind * wa(2) * ef2(x,y) / rho(x,y)
			hlpg3(x,y) = 3._fp_kind * wa(3) * ef3(x,y) / rho(x,y)
			hlpg4(x,y) = 3._fp_kind * wa(4) * ef4(x,y) / rho(x,y)
			hlpg5(x,y) = 3._fp_kind * wa(5) * ef5(x,y) / rho(x,y)
			hlpg6(x,y) = 3._fp_kind * wa(6) * ef6(x,y) / rho(x,y)
			hlpg7(x,y) = 3._fp_kind * wa(7) * ef7(x,y) / rho(x,y)
			hlpg8(x,y) = 3._fp_kind * wa(8) * ef8(x,y) / rho(x,y)
			

			geq0(x,y) = p(x,y) * wa(0) + ga_wa0(x,y) - 0.5_fp_kind * hlpg0(x,y)
			geq1(x,y) = p(x,y) * wa(1) + ga_wa1(x,y) - 0.5_fp_kind * hlpg1(x,y)
			geq2(x,y) = p(x,y) * wa(2) + ga_wa2(x,y) - 0.5_fp_kind * hlpg2(x,y)
			geq3(x,y) = p(x,y) * wa(3) + ga_wa3(x,y) - 0.5_fp_kind * hlpg3(x,y)
			geq4(x,y) = p(x,y) * wa(4) + ga_wa4(x,y) - 0.5_fp_kind * hlpg4(x,y)
			geq5(x,y) = p(x,y) * wa(5) + ga_wa5(x,y) - 0.5_fp_kind * hlpg5(x,y)
			geq6(x,y) = p(x,y) * wa(6) + ga_wa6(x,y) - 0.5_fp_kind * hlpg6(x,y)
			geq7(x,y) = p(x,y) * wa(7) + ga_wa7(x,y) - 0.5_fp_kind * hlpg7(x,y)
			geq8(x,y) = p(x,y) * wa(8) + ga_wa8(x,y) - 0.5_fp_kind * hlpg8(x,y)
			
		


			if (is_solid_node(x,y)==0) then

				g(x,y,0) = g(x,y,0) * (1._fp_kind-(1._fp_kind/tauu(x,y))) + geq0(x,y) * (1._fp_kind/tauu(x,y)) +  hlpg0(x,y)
				g(x,y,1) = g(x,y,1) * (1._fp_kind-(1._fp_kind/tauu(x,y))) + geq1(x,y) * (1._fp_kind/tauu(x,y)) +  hlpg1(x,y)
				g(x,y,2) = g(x,y,2) * (1._fp_kind-(1._fp_kind/tauu(x,y))) + geq2(x,y) * (1._fp_kind/tauu(x,y)) +  hlpg2(x,y)
				g(x,y,3) = g(x,y,3) * (1._fp_kind-(1._fp_kind/tauu(x,y))) + geq3(x,y) * (1._fp_kind/tauu(x,y)) +  hlpg3(x,y)
				g(x,y,4) = g(x,y,4) * (1._fp_kind-(1._fp_kind/tauu(x,y))) + geq4(x,y) * (1._fp_kind/tauu(x,y)) +  hlpg4(x,y)
				g(x,y,5) = g(x,y,5) * (1._fp_kind-(1._fp_kind/tauu(x,y))) + geq5(x,y) * (1._fp_kind/tauu(x,y)) +  hlpg5(x,y)
				g(x,y,6) = g(x,y,6) * (1._fp_kind-(1._fp_kind/tauu(x,y))) + geq6(x,y) * (1._fp_kind/tauu(x,y)) +  hlpg6(x,y)
				g(x,y,7) = g(x,y,7) * (1._fp_kind-(1._fp_kind/tauu(x,y))) + geq7(x,y) * (1._fp_kind/tauu(x,y)) +  hlpg7(x,y)
				g(x,y,8) = g(x,y,8) * (1._fp_kind-(1._fp_kind/tauu(x,y))) + geq8(x,y) * (1._fp_kind/tauu(x,y)) +  hlpg8(x,y)

			end if
					
					
		end if
		
		
							
	end

	attributes(global) subroutine periodic_f
		use precision_m
		use device_var, only : nxd, nyd, h1, h2, g
		implicit none
		integer :: x, y, n(2)
		
		x = (blockIdx%x)*blockDim%x + threadIdx%x 
		y = (blockIdx%y)*blockDim%y + threadIdx%y 
		n(1) = nxd+1!size(g,2) - 1
		n(2) = nyd+1!size(g,3) - 1
		if (x<=n(1) .and. y<=n(2)) then
		!if (x<=n(1) .and. x>=1 .and. y>=1 .and.  y<=n(2)) then
		
		
			! !-- left and right boundaries
			g(0,y,0)=g(nxd,y,0) 
			g(nxd+1,y,0)=g(1,y,0) 
			! !-- bottom and top boundaries
			g(x,0,0) = g(x,nyd,0) 
			g(x,nyd+1,0) = g(x,1,0)

			! !-- left and right boundaries
			g(0,y,1)=g(nxd,y,1) 
			g(nxd+1,y,1)=g(1,y,1) 
			! !-- bottom and top boundaries
			g(x,0,1) = g(x,nyd,1) 
			g(x,nyd+1,1) = g(x,1,1) 

			! !-- left and right boundaries
			g(0,y,2)=g(nxd,y,2) 
			g(nxd+1,y,2)=g(1,y,2) 
			! !-- bottom and top boundaries
			g(x,0,2) = g(x,nyd,2) 
			g(x,nyd+1,2) = g(x,1,2) 	

			! !-- left and right boundaries
			g(0,y,3)=g(nxd,y,3) 
			g(nxd+1,y,3)=g(1,y,3) 
			! !-- bottom and top boundaries
			g(x,0,3) = g(x,nyd,3) 
			g(x,nyd+1,3) = g(x,1,3) 	

			! !-- left and right boundaries
			g(0,y,4)=g(nxd,y,4) 
			g(nxd+1,y,4)=g(1,y,4) 
			! !-- bottom and top boundaries
			g(x,0,4) = g(x,nyd,4) 
			g(x,nyd+1,4) = g(x,1,4) 	

			! !-- left and right boundaries
			g(0,y,5)=g(nxd,y,5) 
			g(nxd+1,y,5)=g(1,y,5) 
			! !-- bottom and top boundaries
			g(x,0,5) = g(x,nyd,5) 
			g(x,nyd+1,5) = g(x,1,5) 	

			! !-- left and right boundaries
			g(0,y,6)=g(nxd,y,6) 
			g(nxd+1,y,6)=g(1,y,6) 
			! !-- bottom and top boundaries
			g(x,0,6) = g(x,nyd,6) 
			g(x,nyd+1,6) = g(x,1,6) 	

			! !-- left and right boundaries
			g(0,y,7)=g(nxd,y,7) 
			g(nxd+1,y,7)=g(1,y,7) 
			! !-- bottom and top boundaries
			g(x,0,7) = g(x,nyd,7) 
			g(x,nyd+1,7) = g(x,1,7) 	

			! !-- left and right boundaries
			g(0,y,8)=g(nxd,y,8) 
			g(nxd+1,y,8)=g(1,y,8) 
			! !-- bottom and top boundaries
			g(x,0,8) = g(x,nyd,8) 
			g(x,nyd+1,8) = g(x,1,8) 				
		


			
			!-- left and right boundaries
			h1(0  ,y,0)=h1(nxd,y,0) 
			h1(nxd+1,y,0)=h1(1 ,y,0) 
			!-- bottom and top boundaries
			h1(x, 0 ,0)=h1(x,nyd,0) 
			h1(x,nyd+1,0)=h1(x,1 ,0) 
			
			!-- left and right boundaries
			h1(0  ,y,1)=h1(nxd,y,1) 
			h1(nxd+1,y,1)=h1(1 ,y,1) 
			!-- bottom and top boundaries
			h1(x, 0 ,1)=h1(x,nyd,1) 
			h1(x,nyd+1,1)=h1(x,1 ,1) 
			
			!-- left and right boundaries
			h1(0  ,y,2)=h1(nxd,y,2) 
			h1(nxd+1,y,2)=h1(1 ,y,2) 
			!-- bottom and top boundaries
			h1(x, 0 ,2)=h1(x,nyd,2) 
			h1(x,nyd+1,2)=h1(x,1 ,2) 
			
			!-- left and right boundaries
			h1(0  ,y,3)=h1(nxd,y,3) 
			h1(nxd+1,y,3)=h1(1 ,y,3) 
			!-- bottom and top boundaries
			h1(x, 0 ,3)=h1(x,nyd,3) 
			h1(x,nyd+1,3)=h1(x,1 ,3) 
			
			!-- left and right boundaries
			h1(0  ,y,4)=h1(nxd,y,4) 
			h1(nxd+1,y,4)=h1(1 ,y,4) 
			!-- bottom and top boundaries
			h1(x, 0 ,4)=h1(x,nyd,4) 
			h1(x,nyd+1,4)=h1(x,1 ,4) 
			
			!-- left and right boundaries
			h1(0  ,y,5)=h1(nxd,y,5) 
			h1(nxd+1,y,5)=h1(1 ,y,5) 
			!-- bottom and top boundaries
			h1(x, 0 ,5)=h1(x,nyd,5) 
			h1(x,nyd+1,5)=h1(x,1 ,5) 
			
			!-- left and right boundaries
			h1(0  ,y,6)=h1(nxd,y,6) 
			h1(nxd+1,y,6)=h1(1 ,y,6) 
			!-- bottom and top boundaries
			h1(x, 0 ,6)=h1(x,nyd,6) 
			h1(x,nyd+1,6)=h1(x,1 ,6) 
			
			!-- left and right boundaries
			h1(0  ,y,7)=h1(nxd,y,7) 
			h1(nxd+1,y,7)=h1(1 ,y,7) 
			!-- bottom and top boundaries
			h1(x, 0 ,7)=h1(x,nyd,7) 
			h1(x,nyd+1,7)=h1(x,1 ,7) 
			
			!-- left and right boundaries
			h1(0  ,y,8)=h1(nxd,y,8) 
			h1(nxd+1,y,8)=h1(1 ,y,8) 
			!-- bottom and top boundaries
			h1(x, 0 ,8)=h1(x,nyd,8) 
			h1(x,nyd+1,8)=h1(x,1 ,8) 
			
			
			
			
			
			
			!-- left and right boundaries
			h2(0  ,y,0)=h2(nxd,y,0) 
			h2(nxd+1,y,0)=h2(1 ,y,0) 
			!-- bottom and top boundaries
			h2(x, 0 ,0)=h2(x,nyd,0) 
			h2(x,nyd+1,0)=h2(x,1 ,0) 
			
			!-- left and right boundaries
			h2(0  ,y,1)=h2(nxd,y,1) 
			h2(nxd+1,y,1)=h2(1 ,y,1) 
			!-- bottom and top boundaries
			h2(x, 0 ,1)=h2(x,nyd,1) 
			h2(x,nyd+1,1)=h2(x,1 ,1) 
			
			!-- left and right boundaries
			h2(0  ,y,2)=h2(nxd,y,2) 
			h2(nxd+1,y,2)=h2(1 ,y,2) 
			!-- bottom and top boundaries
			h2(x, 0 ,2)=h2(x,nyd,2) 
			h2(x,nyd+1,2)=h2(x,1 ,2) 
			
			!-- left and right boundaries
			h2(0  ,y,3)=h2(nxd,y,3) 
			h2(nxd+1,y,3)=h2(1 ,y,3) 
			!-- bottom and top boundaries
			h2(x, 0 ,3)=h2(x,nyd,3) 
			h2(x,nyd+1,3)=h2(x,1 ,3) 
			
			!-- left and right boundaries
			h2(0  ,y,4)=h2(nxd,y,4) 
			h2(nxd+1,y,4)=h2(1 ,y,4) 
			!-- bottom and top boundaries
			h2(x, 0 ,4)=h2(x,nyd,4) 
			h2(x,nyd+1,4)=h2(x,1 ,4) 
			
			!-- left and right boundaries
			h2(0  ,y,5)=h2(nxd,y,5) 
			h2(nxd+1,y,5)=h2(1 ,y,5) 
			!-- bottom and top boundaries
			h2(x, 0 ,5)=h2(x,nyd,5) 
			h2(x,nyd+1,5)=h2(x,1 ,5) 
			
			!-- left and right boundaries
			h2(0  ,y,6)=h2(nxd,y,6) 
			h2(nxd+1,y,6)=h2(1 ,y,6) 
			!-- bottom and top boundaries
			h2(x, 0 ,6)=h2(x,nyd,6) 
			h2(x,nyd+1,6)=h2(x,1 ,6) 
			
			!-- left and right boundaries
			h2(0  ,y,7)=h2(nxd,y,7) 
			h2(nxd+1,y,7)=h2(1 ,y,7) 
			!-- bottom and top boundaries
			h2(x, 0 ,7)=h2(x,nyd,7) 
			h2(x,nyd+1,7)=h2(x,1 ,7) 
			
			!-- left and right boundaries
			h2(0  ,y,8)=h2(nxd,y,8) 
			h2(nxd+1,y,8)=h2(1 ,y,8) 
			!-- bottom and top boundaries
			h2(x, 0 ,8)=h2(x,nyd,8) 
			h2(x,nyd+1,8)=h2(x,1 ,8) 

			

		end if

	
	end
	
	attributes(global) subroutine propagation_gather
		use precision_m
		use device_var
		implicit none
		integer :: x, y, n(2)
		
		x = (blockIdx%x-1)*blockDim%x + threadIdx%x
		y = (blockIdx%y-1)*blockDim%y + threadIdx%y
		n(1) = nxd
		n(2) = nyd
		
		if (x<=n(1) .and. x>=1 .and. y>=1 .and.  y<=n(2)) then

			htemp11(x,y) = h1(x-ex(1),y-ey(1),1)
			htemp12(x,y) = h1(x-ex(2),y-ey(2),2)
			htemp13(x,y) = h1(x-ex(3),y-ey(3),3)
			htemp14(x,y) = h1(x-ex(4),y-ey(4),4)
			htemp15(x,y) = h1(x-ex(5),y-ey(5),5)
			htemp16(x,y) = h1(x-ex(6),y-ey(6),6)
			htemp17(x,y) = h1(x-ex(7),y-ey(7),7)
			htemp18(x,y) = h1(x-ex(8),y-ey(8),8)
			
			htemp21(x,y) = h2(x-ex(1),y-ey(1),1)
			htemp22(x,y) = h2(x-ex(2),y-ey(2),2)
			htemp23(x,y) = h2(x-ex(3),y-ey(3),3)
			htemp24(x,y) = h2(x-ex(4),y-ey(4),4)
			htemp25(x,y) = h2(x-ex(5),y-ey(5),5)
			htemp26(x,y) = h2(x-ex(6),y-ey(6),6)
			htemp27(x,y) = h2(x-ex(7),y-ey(7),7)
			htemp28(x,y) = h2(x-ex(8),y-ey(8),8)

			gtemp1(x,y) = g(x-ex(1),y-ey(1),1)
			gtemp2(x,y) = g(x-ex(2),y-ey(2),2)
			gtemp3(x,y) = g(x-ex(3),y-ey(3),3)
			gtemp4(x,y) = g(x-ex(4),y-ey(4),4)
			gtemp5(x,y) = g(x-ex(5),y-ey(5),5)
			gtemp6(x,y) = g(x-ex(6),y-ey(6),6)
			gtemp7(x,y) = g(x-ex(7),y-ey(7),7)
			gtemp8(x,y) = g(x-ex(8),y-ey(8),8)
			
		end if
		
	end
	
	attributes(global) subroutine propagation_scatter
		use precision_m
		use device_var
		implicit none
		integer :: x, y, n(2)
		
		x = (blockIdx%x-1)*blockDim%x + threadIdx%x
		y = (blockIdx%y-1)*blockDim%y + threadIdx%y
		n(1) = nxd
		n(2) = nyd
		
		if (x<=n(1) .and. x>=1 .and. y>=1 .and.  y<=n(2)) then

			h1(x,y,1) = htemp11(x,y)
			h1(x,y,2) = htemp12(x,y)
			h1(x,y,3) = htemp13(x,y)
			h1(x,y,4) = htemp14(x,y)
			h1(x,y,5) = htemp15(x,y)
			h1(x,y,6) = htemp16(x,y)
			h1(x,y,7) = htemp17(x,y)
			h1(x,y,8) = htemp18(x,y)
			
			h2(x,y,1) = htemp21(x,y)
			h2(x,y,2) = htemp22(x,y)
			h2(x,y,3) = htemp23(x,y)
			h2(x,y,4) = htemp24(x,y)
			h2(x,y,5) = htemp25(x,y)
			h2(x,y,6) = htemp26(x,y)
			h2(x,y,7) = htemp27(x,y)
			h2(x,y,8) = htemp28(x,y)

			g(x,y,1) = gtemp1(x,y)
			g(x,y,2) = gtemp2(x,y)
			g(x,y,3) = gtemp3(x,y)
			g(x,y,4) = gtemp4(x,y)
			g(x,y,5) = gtemp5(x,y)
			g(x,y,6) = gtemp6(x,y)
			g(x,y,7) = gtemp7(x,y)
			g(x,y,8) = gtemp8(x,y)
			
		end if
		
	end
end