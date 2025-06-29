      SUBROUTINE UMAT(STRESS,STATEV,DDSDDE,SSE,SPD,SCD,
     1 RPL,DDSDDT,DRPLDE,DRPLDT,
     2 STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,PREDEF,DPRED,CMNAME,
     3 NDI,NSHR,NTENS,NSTATV,PROPS,NPROPS,COORDS,DROT,PNEWDT,
     4 CELENT,DFGRD0,DFGRD1,NOEL,NPT,LAYER,KSPT,JSTEP,KINC)
C
      INCLUDE 'ABA_PARAM.INC'
C
      CHARACTER*80 CMNAME
      DIMENSION STRESS(NTENS),STATEV(NSTATV),
     1 DDSDDE(NTENS,NTENS),
     2 DDSDDT(NTENS),DRPLDE(NTENS),
     3 STRAN(NTENS),DSTRAN(NTENS),TIME(2),PREDEF(1),DPRED(1),
     4 PROPS(NPROPS),COORDS(3),DROT(3,3),DFGRD0(3,3),DFGRD1(3,3),
     5 JSTEP(4)
      DIMENSION DSTRES(6),D(3,3)

      parameter(three=3.d0,two_third=2.d0/3.d0,four=4.d0,Pi=3.1415926d0)
      parameter(Nele=5760)!Total number of units,different models need to be re-assigned
      parameter(n_points=27) ! Number of integration points in one element

      REAL*8 UVAR(Nele,n_points,20)
      common /share_data/UVAR
      
      stress=1e-15
      DDSDDE=1e-15
     
      
      return
      end subroutine umat

      
      Subroutine matrix_multi(A,B,C,m,n,k)
          Implicit none
          integer m,n,k
          Real*8 A(m,n),B(n,k),C(m,k),temp
          integer i,j,l
          DO i=1,m
              DO j=1,k
                  temp=0.0d0
                  DO l=1,n 				   
                      temp = temp+ A(i,l)*B(l,j)
                  ENDDO
                  C(i,j)=temp
              ENDDO
          ENDDO
      End Subroutine matrix_multi
      
      subroutine get_miu_R(B_p,mR,miu_R)
      implicitnone
      integer i,j,k,l,m,n
      real*8 B_p(3,3),miu_R,lamda_p,x,mR(2)
      lamda_p=sqrt((B_p(1,1)+B_p(2,2)+B_p(3,3))/3.0d0)
      x=lamda_p/mR(2)
      miu_R=mR(1)*x*(3.0d0-x**2.0d0)/(lamda_p*(1.0d0-x**2.0d0))
      end subroutine get_miu_R
      
      
      
      Subroutine mapminmax(x,y,nx,map_form,Iswitch,max_input,min_input,
     1 mean_input,mad_input,layer_node)
      ! Subroutine purpose : data normalization
      ! Iswitch: 1,input; 2,target
      ! map x to [0,1]
      Implicit none
      integer nx,i,Iswitch,map_form
      real*8 x(2),y(2)
      real*8 max_input(2),min_input(2),mean_input(2),mad_input(2)
      real*8 mean_targets(78),mad_targets(78)
      real*8 weight(78,78,4),basis(78,4)
      integer ltype(5),layer_node(5),layer_num
      !print *, 'max_input=',max_input
      if(map_form.eq.1)then
       if(Iswitch .eq. 1) then
          do i = 1,layer_node(1)
              y(i) = 2.0d0*(x(i)-min_input(i))/
     1            (max_input(i)-min_input(i))-1.0d0
          enddo
       elseif(Iswitch .eq. 2) then
          do i = 1,layer_node(layer_num)
              y(i) = (x(i)-mean_targets(i))/mad_targets(i)
     1            
          enddo
       endif
      elseif(map_form .eq. 2) then
       if(Iswitch .eq. 1) then
          do i = 1,layer_node(1)
              y(i) = (x(i)-min_input(i))/
     1            (max_input(i)-min_input(i))
              y(i) =(y(i)-mean_input(i))/
     1            mad_input(i)
              !print *, 'y(i)=',y(i)
          enddo
          
       elseif(Iswitch .eq. 2) then
          do i = 1,layer_node(layer_num)
              y(i) = (x(i)-mean_targets(i))/
     1            mad_targets(i)
          enddo
       endif 
      endif
          
      End Subroutine mapminmax
      
      Subroutine re_mapminmax(x,y,nx,map_form,Iswitch,max_input
     1 ,min_input,mean_input,mad_input,mean_targets,mad_targets,
     1 layer_node)
      ! Subroutine purpose : data opposite normalization
      ! I: 1,input; 2,target
      ! map x from [-1,1] to [min_x,max_x]
      ! y = (x+1.0)*(xmax - xmin)/2.0 +xmin;
      Implicit none
      integer nx,i,Iswitch,map_form
      real*8 max_input(2),min_input(2),mean_input(2),mad_input(2)
      real*8 mean_targets(78),mad_targets(78)
      real*8 weight(78,78,4),basis(78,4)
      integer ltype(5),layer_node(5),layer_num
      real*8 x(nx),y(nx)
      layer_num=5
      if(map_form.eq.1)then
       if(Iswitch .eq. 1) then
          do i = 1,layer_node(1)
              y(i) = (x(i)+1.0d0)*(max_input(i) - min_input(i))
     1              /2.0d0  + min_input(i)
          enddo
       elseif(Iswitch .eq. 2) then
          do i = 1,layer_node(layer_num)
               y(i)=x(i)*(mad_targets(i))+ mean_targets(i)
          enddo
       endif
      elseif(map_form .eq. 2) then
       if(Iswitch .eq. 1) then
          do i = 1,layer_node(1)
              y(i) = x(i)*(max_input(i) - min_input(i))+min_input(i)
          enddo
       elseif(Iswitch .eq. 2) then
          do i = 1,layer_node(layer_num)
            y(i)=x(i)*(mad_targets(i))+ mean_targets(i)    
          enddo
          
       endif 
      endif
      
      End Subroutine re_mapminmax
      
      Subroutine vector_mult_matrix(A,B,C,m,n)
      ! Subroutine purpose : Reture C = A*B
      ! A is a vector, len(A) = m
      ! B is a matrix, size(B) = m*n
      Implicit none
      integer i,j,m,n
      real*8 A(m),B(m,n),C(n)
      do i=1,n
        C(i) = 0.0
      enddo
      do i = 1,m
         do j = 1,n
           C(j) = C(j) + A(i)*B(i,j)
         enddo
      enddo
      End Subroutine vector_mult_matrix
      
      Subroutine activeFCN(A,B,n,ltype)
      ! Subroutine purpose : Reture B = activate(A)
      ! A is a vector, len(A) = m
      ! ltype:: 1:A, 2:tanh(A), 3:leakyReLU
      Implicit none
      integer n,ltype,i
      real*8 A(n),B(n)
      if (ltype==1) then
          do i=1,n
              B(i)=A(i)
          enddo
      elseif(ltype==2) then
          do i=1,n
              B(i)=tanh(A(i))
          enddo
      elseif(ltype==3) then
          do i=1,n
              if (A(i).ge.0.0d0)then
                  B(i)=A(i)
              else
                  B(i)=A(i)*0.01d0
              endif
          enddo
      else
       write(*,*) 'undefine active function'
      endif
      end subroutine activeFCN
      
      Subroutine back_propagation(inputs,weight,basis,layer_node,ltype,
     1 outputs)
      ! Subroutine purpose : Reture output = neural_network(input)
      ! d_inputs:The derivative of the input
      ! input is a vector
      Implicit none
      integer i,j,k,m,n,ninp
      real*8 inputs(2),outputs(78)
      real*8 d_inputs(78,78),d_inputs_s(78,78)
      real*8 A(78),B(78,78),C(78)
      real*8 D(78),E(78),F(78)
      real*8 temp(78,78)
      real*8 d_inps(2,3)
      integer layer_num,layer_node(5),ltype(5),maxnode
      real*8 mean_targets(78),mad_targets(78)
      real*8 weight(78,78,4),basis(78,4)
      maxnode=78
      layer_num=5
	ninp=layer_node(1)
      
      do i=1,ninp
         A(i) = inputs(i)
      enddo
      d_inputs = 0.0d0
      do i=1,maxnode
         d_inputs(i,i) = 1.0
      enddo
c     layer by layer calculation
      do i = 2,layer_num
        do j=1,layer_node(i-1)
           do k=1,layer_node(i)
              B(j,k) = weight(j,k,i-1)
              
           enddo
        enddo
        do j=1,layer_node(i)
           D(j) = basis(j,i-1)
        enddo
        m = layer_node(i-1)
        n = layer_node(i)
        call vector_mult_matrix(A(1:m),B(1:m,1:n),C(1:n),m,n)     !Forward: multiply weight
        E=C+D                                                   !Forward: add bias                                   
        call activeFCN(E(1:n),F(1:n),n,ltype(i))                  !Forward: tansig
        
        do j=1,layer_node(i)
               A(j) = F(j)
        enddo
      enddo
      do i=1,layer_node(layer_num)
          outputs(i) = E(i)
      enddo
      !print *, 'outputs=',outputs
      
      do i=1,ninp
          do j=1,layer_node(layer_num)
              d_inputs(i,j) = temp(i,j)
          enddo
      enddo
      d_inps=d_inputs(1:layer_node(1),1:layer_node(layer_num))
      End Subroutine back_propagation
        
      
      subroutine ANN_recongize(netprops,nprops,max_input,min_input,
     1 mean_input,mad_input,mean_targets,mad_targets,weight,basis,
     1 layer_node,ltype)
      !start with props(3), since props(1) is elastic moduli and props(2) is possion
      implicit none
      integer nprops,n,i,j,k,mn
      real*8 netprops(nprops)
      real*8 max_input(2),min_input(2),mean_input(2),mad_input(2)
      real*8 mean_targets(78),mad_targets(78)
      real*8 weight(78,78,4),basis(78,4)
      integer ltype(5),layer_node(5),layer_num,maxnode
      n=1
      layer_num =int(netprops(n))
      do i=1,layer_num
          n=n+1
          layer_node(i)=int(netprops(n))          !The number of cells in each layer 
          if(i.ne.1)then
              n=n+1
              ltype(i)=int(netprops(n))
          endif
      enddo
      maxnode=maxval(layer_node)
      mn=maxnode
      
      
      do i=1,layer_node(1)
          n = n+1
          max_input(i) = NETPROPS(n)                              !Maximum of input
      enddo
      do i=1,layer_node(1)
          n = n+1
          min_input(i) = NETPROPS(n)                              !Minimum of input
      enddo
      
      
      do i=1,layer_node(1)
          n = n+1
          mean_input(i) = NETPROPS(n)                              !Maximum of input
      enddo
      do i=1,layer_node(1)
          n = n+1
          mad_input(i) = NETPROPS(n)                              !Minimum of input
      enddo
      
      
      do i=1,layer_node(layer_num)
          n = n+1
          mean_targets(i) = NETPROPS(n)                            !Maximum of target
      enddo
      do i=1,layer_node(layer_num)
          n = n+1
          mad_targets(i) = NETPROPS(n)                            !Minimum of target
      enddo
      do i=2,layer_num
          do k=1,layer_node(i)
              do j=1,layer_node(i-1)
                  n = n+1
                  weight(j,k,i-1) = NETPROPS(n)
              enddo
          enddo
          do k=1,layer_node(i)
              n = n+1
              basis(k,i-1) = NETPROPS(n)
          enddo
      enddo
      
      end subroutine ANN_recongize
      
      
      
      
      subroutine ANN(t,E,miu,length,width,inputs,NETPROPS,Ke)
      
      implicitnone
      real*8 inputs(2),targets(78),d_inputs(2,1)
      real*8 ninputs(2),ntargets(78),d_ninputs(2,1)
      real*8 NETPROPS(20000)
      real*8 max_input(2),min_input(2),mean_input(2),mad_input(2)
      real*8 mean_targets(78),mad_targets(78)
      real*8 weight(78,78,4),basis(78,4)
      real*8 rp,e_p,miu_R,Tm,fANN(3),aa,bb
      real*8 B_p(3,3),mR(2)
      real*8 length,width,ratio,E,v,a,b,miu,t,c
      real*8 Ke(24,24)
      !real*8,allocatable :: max_input(:),min_input(:)
      integer i,j,ii,jj,kk,layer_node(5),ltype(5)
      
      character(len=20)   :: time1,time2
      character(len=8)    :: thedate
      character(len=12)   :: thetime
      
      real*8,DIMENSION(3,4) :: COORDS,temp,COORDS_sort
      real*8 x0,y0,x1,y1,x3,y3
      !open(UNIT=3,file='E:\Program\Temp\UELexam\netinfo_14.txt')
      !open(UNIT=107,file='E:\Program\Temp\UELexam\net-disp.txt')
      call date_and_time(thedate,time1)
!      call SYSTEM_CLOCK(time1)
!      write(*,*) 'time1--',time1
!      call ANN_recongize(netprops,5000,max_input,min_input,mean_input,
      call ANN_recongize(NETPROPS(4:11481),11478,max_input,min_input,
     1 mean_input,mad_input,mean_targets,mad_targets,weight,basis,
     1 layer_node,ltype)
      !print *, 'max_input=',max_input
      !print *, 'min_input=',min_input
      !print *, 'mean_input=',mean_input
      !print *, 'mad_input=',mad_input
      !print *, 'mean_targets=',mean_targets
      !print *, 'mad_targets=',mad_targets
      !print *, 'ok'
c----------------------------------------------------
      call mapminmax(inputs,ninputs,size(inputs),2,1,max_input,
     1 min_input,mean_input,mad_input,layer_node)
      !print *, 'inputs=',inputs
c----------------------------------------------------
      call back_propagation(ninputs,weight,basis,layer_node,ltype,
     1 ntargets)
      !print *, 'ntargets=',ntargets
c----------------------------------------------------
      call re_mapminmax(ntargets,targets,size(ntargets),2,2,max_input,
     1 min_input,mean_input,mad_input,mean_targets,mad_targets,
     1 layer_node)
      
c----------------------------------------------------  
      !3 independent variables of the rectangular element stiffness matrix
      !print *, 'targets=',targets
      
      Ke(11,11)=targets(1)
      Ke(11,12)=targets(2)
      Ke(11,15)=targets(3)
      Ke(11,16)=targets(4)
      Ke(11,17)=targets(5)
      Ke(11,18)=targets(6)
      Ke(11,19)=targets(7)
      Ke(11,20)=targets(8)
      Ke(11,21)=targets(9)
      Ke(11,22)=targets(10)
      Ke(11,23)=targets(11)
      Ke(11,24)=targets(12)
      Ke(12,12)=targets(13)
      Ke(12,15)=targets(14)
      Ke(12,16)=targets(15)
      Ke(12,17)=targets(16)
      Ke(12,18)=targets(17)
      Ke(12,19)=targets(18)
      Ke(12,20)=targets(19)
      Ke(12,21)=targets(20)
      Ke(12,22)=targets(21)
      Ke(12,23)=targets(22)
      Ke(12,24)=targets(23)
      Ke(15,15)=targets(24)
      Ke(15,16)=targets(25)
      Ke(15,17)=targets(26)
      Ke(15,18)=targets(27)
      Ke(15,19)=targets(28)
      Ke(15,20)=targets(29)
      Ke(15,21)=targets(30)
      Ke(15,22)=targets(31)
      Ke(15,23)=targets(32)
      Ke(15,24)=targets(33)
      Ke(16,16)=targets(34)
      Ke(16,17)=targets(35)
      Ke(16,18)=targets(36)
      Ke(16,19)=targets(37)
      Ke(16,20)=targets(38)
      Ke(16,21)=targets(39)
      Ke(16,22)=targets(40)
      Ke(16,23)=targets(41)
      Ke(16,24)=targets(42)
      Ke(17,17)=targets(43)
      Ke(17,18)=targets(44)
      Ke(17,19)=targets(45)
      Ke(17,20)=targets(46)
      Ke(17,21)=targets(47)
      Ke(17,22)=targets(48)
      Ke(17,23)=targets(49)
      Ke(17,24)=targets(50)
      Ke(18,18)=targets(51)
      Ke(18,19)=targets(52)
      Ke(18,20)=targets(53)
      Ke(18,21)=targets(54)
      Ke(18,22)=targets(55)
      Ke(18,23)=targets(56)
      Ke(18,24)=targets(57)
      Ke(19,19)=targets(58)
      Ke(19,20)=targets(59)
      Ke(19,21)=targets(60)
      Ke(19,22)=targets(61)
      Ke(19,23)=targets(62)
      Ke(19,24)=targets(63)
      Ke(20,20)=targets(64)
      Ke(20,21)=targets(65)
      Ke(20,22)=targets(66)
      Ke(20,23)=targets(67)
      Ke(20,24)=targets(68)
      Ke(21,21)=targets(69)
      Ke(21,22)=targets(70)
      Ke(21,23)=targets(71)
      Ke(21,24)=targets(72)
      Ke(22,22)=targets(73)
      Ke(22,23)=targets(74)
      Ke(22,24)=targets(75)
      Ke(23,23)=targets(76)
      Ke(23,24)=targets(77)
      Ke(24,24)=targets(78)
      
      
      
      
c---------------------------------------------------- 
      !Remaining single rigid parameters\
      a=0.5*length
      b=0.5*width
      v=miu
      c=t  
      
      Ke(1,1) = a**2*Ke(18,18)/c**2 + 2.0*a**2*Ke(18,21)/c**2 
     1+ a**2*Ke(21,21)/c**2 - 2.0*a**2*Ke(18,20)/(b*c) 
     1- 2.0*a**2*Ke(18,23)/(b*c) - 2.0*a**2*Ke(20,21)/(b*c) 
     1- 2.0*a**2*Ke(21,23)/(b*c) + a**2*Ke(20,20)/b**2 
     1+ 2.0*a**2*Ke(20,23)/b**2 + a**2*Ke(23,23)/b**2-2.0*a*Ke(16,18)/c
     1- 2.0*a*Ke(16,21)/c + 2.0*a*Ke(18,19)/c + 2.0*a*Ke(18,22)/c 
     1+ 2.0*a*Ke(19,21)/c + 2.0*a*Ke(21,22)/c + 2.0*a*Ke(16,20)/b 
     1+ 2.0*a*Ke(16,23)/b - 2.0*a*Ke(19,20)/b - 2.0*a*Ke(19,23)/b 
     1- 2.0*a*Ke(20,22)/b - 2.0*a*Ke(22,23)/b + Ke(16,16) -2.0*Ke(16,19)
     1- 2.0*Ke(16,22) + Ke(19,19) + 2.0*Ke(19,22) + Ke(22,22) 
     1+ (-E*a**2*c*v + E*a**2*c + 2.0*E*b**2*c)/(6.0*a*b*v**2 - 6.0*a*b)
      Ke(1,2) = -a**3*Ke(15,18)/(b*c**2) - a**3*Ke(15,21)/(b*c**2) 
     1- a**3*Ke(18,18)/(b*c**2) - 2.0*a**3*Ke(18,21)/(b*c**2) 
     1- a**3*Ke(18,24)/(b*c**2) - a**3*Ke(21,21)/(b*c**2) 
     1- a**3*Ke(21,24)/(b*c**2) + a**3*Ke(15,20)/(b**2*c) 
     1+ a**3*Ke(15,23)/(b**2*c) + a**3*Ke(18,23)/(b**2*c)
     1+ a**3*Ke(20,24)/(b**2*c) + a**3*Ke(21,23)/(b**2*c) 
     1+ a**3*Ke(23,24)/(b**2*c) + a**2*Ke(15,16)/(b*c) - a**2*Ke(15,19)
     1/(b*c) - a**2*Ke(15,22)/(b*c) + a**2*Ke(16,18)/(b*c) 
     1+ a**2*Ke(16,21)/(b*c) + a**2*Ke(16,24)/(b*c) - 3.0*a**2*Ke(18,19)
     1/(b*c) - 3.0*a**2*Ke(18,22)/(b*c) - 3.0*a**2*Ke(19,21)/(b*c) 
     1- a**2*Ke(19,24)/(b*c) - 3.0*a**2*Ke(21,22)/(b*c) - a**2*Ke(22,24)
     1/(b*c) + 2.0*a**2*Ke(19,23)/b**2 + 2.0*a**2*Ke(22,23)/b**2 
     1+ a*Ke(11,18)/c + a*Ke(11,21)/c - a*Ke(17,18)/c - a*Ke(17,21)/c 
     1- a*Ke(11,20)/b - a*Ke(11,23)/b + 2.0*a*Ke(16,19)/b 
     1+ 2.0*a*Ke(16,22)/b + a*Ke(17,20)/b + a*Ke(17,23)/b 
     1- 2.0*a*Ke(19,19)/b - 4.0*a*Ke(19,22)/b + a*Ke(20,20)/b 
     1+ a*Ke(20,23)/b - 2.0*a*Ke(22,22)/b - Ke(11,16) + Ke(11,19) 
     1+ Ke(11,22) + Ke(16,17) + Ke(16,20) - Ke(17,19) - Ke(17,22) 
     1+ (E*c*v - 2.0*E*c)/(3.0*v**2 - 3.0) + Ke(19,20)*(2.0*a**2 
     1- b**2)/b**2 + Ke(20,22)*(2.0*a**2 - b**2)/b**2 + Ke(18,20)
     1*(a**3 - a*b**2)/(b**2*c) + Ke(20,21)*(a**3 - a*b**2)/(b**2*c)
      Ke(1,3) = E*c**2/(b*v + b) - a**3*Ke(18,18)/(b**2*c) 
     1- 2.0*a**3*Ke(18,21)/(b**2*c) - a**3*Ke(21,21)/(b**2*c) 
     1+ a**3*Ke(18,20)/b**3 + a**3*Ke(20,21)/b**3 + a**2*Ke(16,18)
     1 /b**2 + a**2*Ke(16,21)/b**2 - 3.0*a**2*Ke(18,19)/b**2 
     1 - 3.0*a**2*Ke(18,22)/b**2 - 3.0*a**2*Ke(19,21)/b**2 
     1- 3.0*a**2*Ke(21,22)/b**2 + 2.0*a**2*c*Ke(19,20)/b**3 
     1+ 2.0*a**2*c*Ke(20,22)/b**3 + a*Ke(12,18)/c + a*Ke(12,21)/c 
     1+ 2.0*a*Ke(11,18)/b + 2.0*a*Ke(11,21)/b - a*Ke(12,20)/b 
     1 - a*Ke(12,23)/b - 2.0*a*c*Ke(11,20)/b**2 - 2.0*a*c*Ke(11,23)
     1/b**2 + 2.0*a*c*Ke(16,19)/b**2 + 2.0*a*c*Ke(16,22)/b**2 
     1 - 2.0*a*c*Ke(19,19)/b**2 - 4.0*a*c*Ke(19,22)/b**2 
     1- 2.0*a*c*Ke(20,23)/b**2 - 2.0*a*c*Ke(22,22)/b**2 
     1- 2.0*a*c*Ke(23,23)/b**2 - Ke(12,16) + Ke(12,19) + Ke(12,22) 
     1 - 2.0*c*Ke(11,16)/b + 2.0*c*Ke(11,19)/b + 2.0*c*Ke(11,22)/b 
     1 - 2.0*c*Ke(16,23)/b + Ke(15,16)*(a**2 - b**2)/b**2 + Ke(15,19)
     1*(-a**2 + b**2)/b**2 + Ke(15,22)*(-a**2 + b**2)/b**2 + Ke(16,24)
     1*(a**2 - b**2)/b**2 + Ke(19,24)*(-a**2 + b**2)/b**2 + Ke(22,24)
     1*(-a**2 + b**2)/b**2 + Ke(15,18)*(-a**3 + a*b**2)/(b**2*c) 
     1+ Ke(15,21)*(-a**3 + a*b**2)/(b**2*c) + Ke(18,24)*(-a**3 + a*b**2)
     1/(b**2*c) + Ke(21,24)*(-a**3 + a*b**2)/(b**2*c) + Ke(15,20)*(a**3
     1- a*b**2)/b**3 + Ke(15,23)*(a**3 - a*b**2)/b**3 + Ke(18,23)*(a**3 
     1+ 2.0*a*b**2)/b**3 + Ke(19,23)*(2.0*a**2*c + 2.0*b**2*c)/b**3 
     1+ Ke(20,24)*(a**3 - a*b**2)/b**3 + Ke(21,23)*(a**3 + 2.0*a*b**2)
     1/b**3 + Ke(22,23)*(2.0*a**2*c + 2.0*b**2*c)/b**3 
     1+ Ke(23,24)*(a**3 - a*b**2)/b**3
      Ke(1,4) = -a**2*Ke(18,18)/c**2 - 2.0*a**2*Ke(18,21)/c**2 
     1- a**2*Ke(21,21)/c**2 + 2.0*a**2*Ke(18,20)/(b*c) 
     1+ 2.0*a**2*Ke(18,23)/(b*c) + 2.0*a**2*Ke(20,21)/(b*c) 
     1+ 2.0*a**2*Ke(21,23)/(b*c) - a**2*Ke(20,20)/b**2 
     1- 2.0*a**2*Ke(20,23)/b**2 - a**2*Ke(23,23)/b**2 + 2.0*a*Ke(16,18)
     1/c + 2.0*a*Ke(16,21)/c - a*Ke(18,19)/c - a*Ke(18,22)/c 
     1- a*Ke(19,21)/c - a*Ke(21,22)/c - 2.0*a*Ke(16,20)/b 
     1- 2.0*a*Ke(16,23)/b + a*Ke(19,20)/b + a*Ke(19,23)/b 
     1+ a*Ke(20,22)/b + a*Ke(22,23)/b - Ke(16,16) + Ke(16,19) 
     1+ Ke(16,22) + (E*a**2*c*v - E*a**2*c - E*b**2*c)
     1/(6.0*a*b*v**2 - 6.0*a*b)
      Ke(1,5) = a*Ke(11,18)/c + a*Ke(11,21)/c + a*Ke(17,18)/c 
     1+ a*Ke(17,21)/c + a*Ke(18,23)/c + a*Ke(21,23)/c - a*Ke(11,20)/b 
     1- a*Ke(11,23)/b - a*Ke(17,20)/b - a*Ke(17,23)/b - a*Ke(20,23)/b 
     1- a*Ke(23,23)/b - Ke(11,16) + Ke(11,19) + Ke(11,22) - Ke(16,17) 
     1- Ke(16,23) + Ke(17,19) + Ke(17,22) + Ke(19,23) + Ke(22,23) 
     1+ (3.0*E*c*v + E*c)/(12.0*v**2 - 12.0)
      Ke(1,6) = -a*Ke(12,18)/c - a*Ke(12,21)/c - 2.0*a*Ke(11,18)/b 
     1- 2.0*a*Ke(11,21)/b + a*Ke(12,20)/b + a*Ke(12,23)/b 
     1+ 2.0*a*c*Ke(11,20)/b**2 + 2.0*a*c*Ke(11,23)/b**2 
     1+ 2.0*a*c*Ke(20,23)/b**2 + 2.0*a*c*Ke(23,23)/b**2 + Ke(12,16) 
     1- Ke(12,19) - Ke(12,22) + (-E*b**2 - E*c**2*v + E*c**2)/(b*v**2 
     1- b) + 2.0*c*Ke(11,16)/b - 2.0*c*Ke(11,19)/b - 2.0*c*Ke(11,22)/b 
     1+ 2.0*c*Ke(16,23)/b + Ke(16,19)*(2.0*a*b**2 - 2.0*a*c**2)/(b**2*c)
     1+ Ke(16,22)*(2.0*a*b**2 - 2.0*a*c**2)/(b**2*c) + Ke(19,19)
     1*(-2.0*a*b**2 + 2.0*a*c**2)/(b**2*c) + Ke(19,22)*(-4.0*a*b**2 
     1+ 4.0*a*c**2)/(b**2*c) + Ke(22,22)*(-2.0*a*b**2 + 2.0*a*c**2)
     1/(b**2*c) + Ke(15,16)*(a**2*b**2 - a**2*c**2)/(b**2*c**2) 
     1+ Ke(15,19)*(-a**2*b**2 + a**2*c**2)/(b**2*c**2) + Ke(15,22)
     1*(-a**2*b**2 + a**2*c**2)/(b**2*c**2) + Ke(16,18)*(a**2*b**2 
     1- a**2*c**2 - b**2*c**2)/(b**2*c**2) + Ke(16,21)*(a**2*b**2 
     1- a**2*c**2)/(b**2*c**2) + Ke(16,24)*(a**2*b**2 - a**2*c**2 
     1+ b**2*c**2)/(b**2*c**2) + Ke(18,19)*(-3.0*a**2*b**2 
     1+ 3.0*a**2*c**2 + b**2*c**2)/(b**2*c**2) + Ke(18,22)
     1*(-3.0*a**2*b**2 + 3.0*a**2*c**2 + b**2*c**2)/(b**2*c**2) 
     1+ Ke(19,21)*(-3.0*a**2*b**2 + 3.0*a**2*c**2)/(b**2*c**2) 
     1+ Ke(19,24)*(-a**2*b**2 + a**2*c**2 - b**2*c**2)/(b**2*c**2) 
     1+ Ke(21,22)*(-3.0*a**2*b**2 + 3.0*a**2*c**2)/(b**2*c**2) 
     1+ Ke(22,24)*(-a**2*b**2 + a**2*c**2 - b**2*c**2)/(b**2*c**2) 
     1+ Ke(15,18)*(-a**3*b**2 + a**3*c**2)/(b**2*c**3) 
     1+ Ke(15,21)*(-a**3*b**2 + a**3*c**2)/(b**2*c**3) 
     1+ Ke(18,18)*(-a**3*b**2 + a**3*c**2 + a*b**2*c**2)/(b**2*c**3) 
     1+ Ke(18,21)*(-2.0*a**3*b**2 + 2.0*a**3*c**2 + a*b**2*c**2)
     1/(b**2*c**3) + Ke(18,24)*(-a**3*b**2 + a**3*c**2 - a*b**2*c**2)
     1/(b**2*c**3) + Ke(21,21)*(-a**3*b**2 + a**3*c**2)/(b**2*c**3) 
     1+ Ke(21,24)*(-a**3*b**2 + a**3*c**2 - a*b**2*c**2)/(b**2*c**3) 
     1+ Ke(19,20)*(2.0*a**2*b**2 - 2.0*a**2*c**2)/(b**3*c) 
     1+ Ke(19,23)*(2.0*a**2*b**2 - 2.0*a**2*c**2 - 2.0*b**2*c**2)
     1/(b**3*c) + Ke(20,22)*(2.0*a**2*b**2 - 2.0*a**2*c**2)/(b**3*c) 
     1+ Ke(22,23)*(2.0*a**2*b**2 - 2.0*a**2*c**2 - 2.0*b**2*c**2)
     1/(b**3*c) + Ke(15,20)*(a**3*b**2 - a**3*c**2)/(b**3*c**2) 
     1+ Ke(15,23)*(a**3*b**2 - a**3*c**2)/(b**3*c**2) + Ke(18,20)
     1*(a**3*b**2 - a**3*c**2 - a*b**2*c**2)/(b**3*c**2) + Ke(18,23)
     1*(a**3*b**2 - a**3*c**2 - 3.0*a*b**2*c**2)/(b**3*c**2) 
     1+ Ke(20,21)*(a**3*b**2 - a**3*c**2)/(b**3*c**2) + Ke(20,24)
     1*(a**3*b**2 - a**3*c**2 + a*b**2*c**2)/(b**3*c**2) + Ke(21,23)
     1*(a**3*b**2 - a**3*c**2 - 2.0*a*b**2*c**2)/(b**3*c**2) 
     1+ Ke(23,24)*(a**3*b**2 - a**3*c**2 + a*b**2*c**2)/(b**3*c**2)
      Ke(1,7)= a**2*Ke(18,18)/c**2 + 2.0*a**2*Ke(18,21)/c**2 
     1+ a**2*Ke(21,21)/c**2 - 2.0*a**2*Ke(18,20)/(b*c) 
     1- 2.0*a**2*Ke(18,23)/(b*c) - 2.0*a**2*Ke(20,21)/(b*c) 
     1- 2.0*a**2*Ke(21,23)/(b*c) + a**2*Ke(20,20)/b**2 
     1+ 2.0*a**2*Ke(20,23)/b**2 + a**2*Ke(23,23)/b**2 - a*Ke(16,18)/c 
     1- a*Ke(16,21)/c + 2.0*a*Ke(18,19)/c + a*Ke(18,22)/c 
     1+ 2.0*a*Ke(19,21)/c + a*Ke(21,22)/c + a*Ke(16,20)/b+a*Ke(16,23)/b
     1- 2.0*a*Ke(19,20)/b - 2.0*a*Ke(19,23)/b - a*Ke(20,22)/b 
     1- a*Ke(22,23)/b - Ke(16,19) + Ke(19,19) + Ke(19,22) 
     1+ (-5.0*E*a**2*c*v + 5.0*E*a**2*c + 2.0*E*b**2*c)
     1/(12.0*a*b*v**2 - 12.0*a*b)
      Ke(1,8) = a**3*Ke(15,18)/(b*c**2) + a**3*Ke(15,21)/(b*c**2) 
     1+ a**3*Ke(18,18)/(b*c**2) + 2.0*a**3*Ke(18,21)/(b*c**2) 
     1+ a**3*Ke(18,24)/(b*c**2) + a**3*Ke(21,21)/(b*c**2) 
     1+ a**3*Ke(21,24)/(b*c**2) - a**3*Ke(15,20)/(b**2*c) 
     1- a**3*Ke(15,23)/(b**2*c) - a**3*Ke(20,24)/(b**2*c) 
     1- a**3*Ke(23,24)/(b**2*c) - a**2*Ke(15,16)/(b*c) + a**2*Ke(15,19)
     1/(b*c) + a**2*Ke(15,22)/(b*c) - a**2*Ke(16,18)/(b*c) 
     1- a**2*Ke(16,21)/(b*c) - a**2*Ke(16,24)/(b*c) + 3.0*a**2*Ke(18,19)
     1/(b*c) + 3.0*a**2*Ke(18,22)/(b*c) + 3.0*a**2*Ke(19,21)/(b*c) 
     1+ a**2*Ke(19,24)/(b*c) + 3.0*a**2*Ke(21,22)/(b*c) + a**2*Ke(22,24)
     1/(b*c) - a*Ke(11,18)/c - a*Ke(11,21)/c + a*Ke(11,20)/b 
     1+ a*Ke(11,23)/b - 2.0*a*Ke(16,19)/b - 2.0*a*Ke(16,22)/b 
     1+ 2.0*a*Ke(19,19)/b + 4.0*a*Ke(19,22)/b - a*Ke(20,20)/b 
     1+ 2.0*a*Ke(22,22)/b + a*Ke(23,23)/b + Ke(11,16) - Ke(11,19) 
     1- Ke(11,22) - Ke(16,20) + Ke(16,23) + (-9.0*E*c*v + 11.0*E*c)
     1/(12.0*v**2 - 12.0) + Ke(19,20)*(-2.0*a**2 + b**2)/b**2 
     1+ Ke(19,23)*(-2.0*a**2 - b**2)/b**2 + Ke(20,22)*(-2.0*a**2 
     1+ b**2)/b**2 + Ke(22,23)*(-2.0*a**2 - b**2)/b**2 + Ke(18,20)
     1*(-a**3 + a*b**2)/(b**2*c) + Ke(18,23)*(-a**3 - a*b**2)/(b**2*c) 
     1+ Ke(20,21)*(-a**3 + a*b**2)/(b**2*c) 
     1+ Ke(21,23)*(-a**3 - a*b**2)/(b**2*c)
      Ke(1,9) = E*b/(v**2 - 1.0) + a**3*Ke(15,18)/c**3 + a**3*Ke(15,21)
     1/c**3 + a**3*Ke(18,18)/c**3 - a**3*Ke(15,20)/(b*c**2) 
     1- a**3*Ke(15,23)/(b*c**2) - a**3*Ke(18,20)/(b*c**2) 
     1- a**3*Ke(18,23)/(b*c**2) - a**2*Ke(15,16)/c**2 + a**2*Ke(15,19)
     1/c**2 + a**2*Ke(15,22)/c**2 - a**2*Ke(16,18)/c**2 
     1+ 3.0*a**2*Ke(18,19)/c**2 + 3.0*a**2*Ke(18,22)/c**2 
     1- 2.0*a**2*Ke(19,20)/(b*c) - 2.0*a**2*Ke(19,23)/(b*c) 
     1- 2.0*a**2*Ke(20,22)/(b*c) - 2.0*a**2*Ke(22,23)/(b*c) 
     1+ a*Ke(12,18)/c + a*Ke(12,21)/c - 2.0*a*Ke(16,19)/c 
     1- 2.0*a*Ke(16,22)/c + 2.0*a*Ke(19,19)/c + 4.0*a*Ke(19,22)/c 
     1+ 2.0*a*Ke(22,22)/c - a*Ke(12,20)/b - a*Ke(12,23)/b - Ke(12,16) 
     1+ Ke(12,19) + Ke(12,22) + Ke(16,21)*(-a**2 - c**2)/c**2 
     1+ Ke(16,24)*(-a**2 - c**2)/c**2 + Ke(19,21)*(3.0*a**2 + c**2)
     1/c**2 + Ke(19,24)*(a**2 + c**2)/c**2 + Ke(21,22)*(3.0*a**2 
     1+ c**2)/c**2 + Ke(22,24)*(a**2 + c**2)/c**2 + Ke(18,21)*(2.0*a**3 
     1+ a*c**2)/c**3 + Ke(18,24)*(a**3 + a*c**2)/c**3 + Ke(21,21)*(a**3 
     1+ a*c**2)/c**3 + Ke(21,24)*(a**3 + a*c**2)/c**3 + Ke(20,21)*(-a**3 
     1- a*c**2)/(b*c**2) + Ke(20,24)*(-a**3 - a*c**2)/(b*c**2) 
     1+ Ke(21,23)*(-a**3 - a*c**2)/(b*c**2) 
     1+ Ke(23,24)*(-a**3 - a*c**2)/(b*c**2)
      Ke(1,10) = a**2*Ke(15,18)/c**2 + a**2*Ke(15,21)/c**2 
     1+ a**2*Ke(18,24)/c**2 + a**2*Ke(21,24)/c**2 - a**2*Ke(15,20)/(b*c)
     1- a**2*Ke(15,23)/(b*c) + a**2*Ke(18,20)/(b*c)+a**2*Ke(18,23)/(b*c)
     1+ a**2*Ke(20,21)/(b*c) - a**2*Ke(20,24)/(b*c)+a**2*Ke(21,23)/(b*c)
     1- a**2*Ke(23,24)/(b*c) - a**2*Ke(20,20)/b**2 - 2.0*a**2*Ke(20,23)
     1/b**2 - a**2*Ke(23,23)/b**2 - a*Ke(15,16)/c + a*Ke(15,19)/c 
     1+ a*Ke(15,22)/c - a*Ke(16,24)/c + a*Ke(18,22)/c + a*Ke(19,24)/c 
     1+ a*Ke(21,22)/c + a*Ke(22,24)/c - a*Ke(16,20)/b - a*Ke(16,23)/b 
     1+ a*Ke(19,20)/b + a*Ke(19,23)/b - Ke(16,22) + Ke(19,22) +Ke(22,22)
     1+ (5.0*E*a**2*c*v - 5.0*E*a**2*c + 2.0*E*b**2*c)
     1/(12.0*a*b*v**2 - 12.0*a*b)
      Ke(1,11) = -E*c/(12.0*v - 12.0) - a*Ke(11,18)/c - a*Ke(11,21)/c 
     1+ a*Ke(11,20)/b + a*Ke(11,23)/b + Ke(11,16) - Ke(11,19) - Ke(11,22)
      Ke(1,12) = -a*Ke(12,18)/c - a*Ke(12,21)/c + a*Ke(12,20)/b 
     1+ a*Ke(12,23)/b + Ke(12,16) - Ke(12,19) - Ke(12,22)
      Ke(1,13) = -a**2*Ke(15,18)/c**2 - a**2*Ke(15,21)/c**2 
     1- a**2*Ke(18,18)/c**2 - 2.0*a**2*Ke(18,21)/c**2 - a**2*Ke(18,24)
     1/c**2 - a**2*Ke(21,21)/c**2 - a**2*Ke(21,24)/c**2 + a**2*Ke(15,20)
     1/(b*c) + a**2*Ke(15,23)/(b*c) + a**2*Ke(18,20)/(b*c) 
     1+ a**2*Ke(18,23)/(b*c) + a**2*Ke(20,21)/(b*c) + a**2*Ke(20,24)
     1/(b*c) + a**2*Ke(21,23)/(b*c) + a**2*Ke(23,24)/(b*c) + a*Ke(15,16)
     1/c - a*Ke(15,19)/c - a*Ke(15,22)/c + 2.0*a*Ke(16,18)/c 
     1+ 2.0*a*Ke(16,21)/c + a*Ke(16,24)/c - 2.0*a*Ke(18,19)/c 
     1- 2.0*a*Ke(18,22)/c - 2.0*a*Ke(19,21)/c - a*Ke(19,24)/c 
     1- 2.0*a*Ke(21,22)/c - a*Ke(22,24)/c - a*Ke(16,20)/b-a*Ke(16,23)/b
     1+ a*Ke(19,20)/b + a*Ke(19,23)/b + a*Ke(20,22)/b + a*Ke(22,23)/b 
     1- Ke(16,16) + 2.0*Ke(16,19) + 2.0*Ke(16,22) - Ke(19,19) 
     1- 2.0*Ke(19,22) - Ke(22,22) + (E*a**2*c*v - E*a**2*c 
     1- 4.0*E*b**2*c)/(12.0*a*b*v**2 - 12.0*a*b)
      Ke(1,14) = a*Ke(17,18)/c + a*Ke(17,21)/c + a*Ke(18,20)/c 
     1+ a*Ke(18,23)/c + a*Ke(20,21)/c + a*Ke(21,23)/c - a*Ke(17,20)/b
     1- a*Ke(17,23)/b - a*Ke(20,20)/b - 2.0*a*Ke(20,23)/b 
     1- a*Ke(23,23)/b - Ke(16,17) - Ke(16,20) - Ke(16,23) + Ke(17,19) 
     1+ Ke(17,22) + Ke(19,20) + Ke(19,23) + Ke(20,22) + Ke(22,23) 
     1+ (3.0*E*c*v - 2.0*E*c)/(6.0*v**2 - 6.0)
      Ke(1,15) = -a*Ke(15,18)/c - a*Ke(15,21)/c + a*Ke(15,20)/b 
     1+ a*Ke(15,23)/b + Ke(15,16) - Ke(15,19) - Ke(15,22)
      Ke(1,16) = -a*Ke(16,18)/c - a*Ke(16,21)/c + a*Ke(16,20)/b 
     1+ a*Ke(16,23)/b + Ke(16,16) - Ke(16,19) - Ke(16,22) 
     1+(-E*a**2*c*v + E*a**2*c + 2.0*E*b**2*c)/(12.0*a*b*v**2-12.0*a*b)
      Ke(1,17) = -E*c/(12.0*v - 12.0) - a*Ke(17,18)/c - a*Ke(17,21)/c 
     1+a*Ke(17,20)/b + a*Ke(17,23)/b + Ke(16,17) - Ke(17,19) - Ke(17,22)
      Ke(1,18)= -a*Ke(18,18)/c - a*Ke(18,21)/c + a*Ke(18,20)/b 
     1+ a*Ke(18,23)/b + Ke(16,18) - Ke(18,19) - Ke(18,22)
      Ke(1,19) = -a*Ke(18,19)/c - a*Ke(19,21)/c + a*Ke(19,20)/b 
     1+ a*Ke(19,23)/b + Ke(16,19) - Ke(19,19) - Ke(19,22) 
     1+(E*a**2*c*v - E*a**2*c - 2.0*E*b**2*c)/(12.0*a*b*v**2 - 12.0*a*b)
      Ke(1,20) = E*c/(12.0*v - 12.0) - a*Ke(18,20)/c - a*Ke(20,21)/c 
     1+ a*Ke(20,20)/b + a*Ke(20,23)/b + Ke(16,20) -Ke(19,20)- Ke(20,22)
      Ke(1,21) = -a*Ke(18,21)/c - a*Ke(21,21)/c + a*Ke(20,21)/b 
     1+ a*Ke(21,23)/b + Ke(16,21) - Ke(19,21) - Ke(21,22)
      Ke(1,22) = -a*Ke(18,22)/c - a*Ke(21,22)/c + a*Ke(20,22)/b 
     1+ a*Ke(22,23)/b + Ke(16,22) - Ke(19,22) - Ke(22,22) 
     1+(-E*a**2*c*v + E*a**2*c - 2.0*E*b**2*c)/(12.0*a*b*v**2 -12.0*a*b)
      Ke(1,23) = -a*Ke(18,23)/c - a*Ke(21,23)/c + a*Ke(20,23)/b 
     1+ a*Ke(23,23)/b + Ke(16,23) - Ke(19,23) - Ke(22,23) 
     1+ (-3.0*E*c*v + E*c)/(12.0*v**2 - 12.0)
      Ke(1,24) = -a*Ke(18,24)/c - a*Ke(21,24)/c + a*Ke(20,24)/b 
     1+ a*Ke(23,24)/b + Ke(16,24) - Ke(19,24) - Ke(22,24)
      Ke(2,2) = a**4*Ke(15,15)/(b**2*c**2) + 2.0*a**4*Ke(15,18)
     1/(b**2*c**2) + 2.0*a**4*Ke(15,21)/(b**2*c**2) + 2.0*a**4*Ke(15,24)
     1/(b**2*c**2) + a**4*Ke(18,18)/(b**2*c**2) + 2.0*a**4*Ke(18,21)
     1/(b**2*c**2) + 2.0*a**4*Ke(18,24)/(b**2*c**2) + a**4*Ke(21,21)
     1/(b**2*c**2) + 2.0*a**4*Ke(21,24)/(b**2*c**2) + a**4*Ke(24,24)
     1/(b**2*c**2) + 4.0*a**3*Ke(15,19)/(b**2*c) + 4.0*a**3*Ke(15,22)
     1/(b**2*c) + 4.0*a**3*Ke(18,19)/(b**2*c) + 4.0*a**3*Ke(18,22)
     1/(b**2*c) + 4.0*a**3*Ke(19,21)/(b**2*c) + 4.0*a**3*Ke(19,24)
     1/(b**2*c) + 4.0*a**3*Ke(21,22)/(b**2*c) + 4.0*a**3*Ke(22,24)
     1/(b**2*c) - 2.0*a**2*Ke(11,15)/(b*c) - 2.0*a**2*Ke(11,18)
     1/(b*c) - 2.0*a**2*Ke(11,21)/(b*c) - 2.0*a**2*Ke(11,24)/(b*c) 
     1+ 2.0*a**2*Ke(15,17)/(b*c) + 2.0*a**2*Ke(15,20)/(b*c) 
     1+ 2.0*a**2*Ke(17,18)/(b*c) + 2.0*a**2*Ke(17,21)/(b*c) 
     1+ 2.0*a**2*Ke(17,24)/(b*c) + 2.0*a**2*Ke(18,20)/(b*c) 
     1+ 2.0*a**2*Ke(20,21)/(b*c) + 2.0*a**2*Ke(20,24)/(b*c) 
     1+ 4.0*a**2*Ke(19,19)/b**2 + 8.0*a**2*Ke(19,22)/b**2 
     1+ 4.0*a**2*Ke(22,22)/b**2 - 4.0*a*Ke(11,19)/b - 4.0*a*Ke(11,22)/b 
     1+ 4.0*a*Ke(17,19)/b + 4.0*a*Ke(17,22)/b + 4.0*a*Ke(19,20)/b 
     1+ 4.0*a*Ke(20,22)/b + Ke(11,11) - 2.0*Ke(11,17) - 2.0*Ke(11,20) 
     1+ Ke(17,17) + 2.0*Ke(17,20) + Ke(20,20) + (-20.0*E*a**2*c*v 
     1+ 22.0*E*a**2*c - 3.0*E*b**2*c*v + 3.0*E*b**2*c)
     1/(6.0*a*b*v**2 - 6.0*a*b)
      Ke(2,3) = -13.0*E*a*c**2/(3.0*b**2*v + 3.0*b**2) + a**4*Ke(18,18)
     1/(b**3*c) + 2.0*a**4*Ke(18,21)/(b**3*c) + a**4*Ke(21,21)/(b**3*c) 
     1+ 4.0*a**3*Ke(18,19)/b**3 + 4.0*a**3*Ke(18,22)/b**3 
     1+ 4.0*a**3*Ke(19,21)/b**3 + 4.0*a**3*Ke(21,22)/b**3 
     1- a**2*Ke(12,15)/(b*c) - a**2*Ke(12,18)/(b*c) 
     1- a**2*Ke(12,21)/(b*c) - a**2*Ke(12,24)/(b*c) 
     1- 3.0*a**2*Ke(11,18)/b**2 - 3.0*a**2*Ke(11,21)/b**2 
     1- 2.0*a**2*Ke(15,23)/b**2 + a**2*Ke(17,18)/b**2 
     1+ a**2*Ke(17,21)/b**2 + a**2*Ke(18,20)/b**2 
     1- 2.0*a**2*Ke(18,23)/b**2 + a**2*Ke(20,21)/b**2 
     1- 2.0*a**2*Ke(21,23)/b**2 - 2.0*a**2*Ke(23,24)/b**2 
     1+ 4.0*a**2*c*Ke(19,19)/b**3 + 8.0*a**2*c*Ke(19,22)/b**3 
     1+ 4.0*a**2*c*Ke(22,22)/b**3 - 2.0*a*Ke(12,19)/b-2.0*a*Ke(12,22)/b
     1- 6.0*a*c*Ke(11,19)/b**2 - 6.0*a*c*Ke(11,22)/b**2 
     1+ 2.0*a*c*Ke(17,19)/b**2 + 2.0*a*c*Ke(17,22)/b**2 
     1+ 2.0*a*c*Ke(19,20)/b**2 - 4.0*a*c*Ke(19,23)/b**2 
     1+ 2.0*a*c*Ke(20,22)/b**2 - 4.0*a*c*Ke(22,23)/b**2 
     1+ Ke(11,12) - Ke(12,17) - Ke(12,20) + 2.0*c*Ke(11,11)/b 
     1- 2.0*c*Ke(11,17)/b - 2.0*c*Ke(11,20)/b + 2.0*c*Ke(11,23)/b 
     1- 2.0*c*Ke(17,23)/b - 2.0*c*Ke(20,23)/b + Ke(11,15)*(-3.0*a**2 
     1+ b**2)/b**2 + Ke(11,24)*(-3.0*a**2 + b**2)/b**2 + Ke(15,17)*(a**2
     1- b**2)/b**2 + Ke(15,20)*(a**2 - b**2)/b**2 + Ke(17,24)*(a**2 
     1- b**2)/b**2 + Ke(20,24)*(a**2 - b**2)/b**2 + Ke(15,19)*(4.0*a**3
     1- 2.0*a*b**2)/b**3 + Ke(15,22)*(4.0*a**3 - 2.0*a*b**2)/b**3 
     1+ Ke(19,24)*(4.0*a**3 - 2.0*a*b**2)/b**3 + Ke(22,24)*(4.0*a**3 
     1- 2.0*a*b**2)/b**3 + Ke(15,15)*(a**4 - a**2*b**2)/(b**3*c) 
     1+ Ke(15,18)*(2.0*a**4 - a**2*b**2)/(b**3*c) + Ke(15,21)*(2.0*a**4 
     1- a**2*b**2)/(b**3*c) + Ke(15,24)*(2.0*a**4 - 2.0*a**2*b**2)
     1/(b**3*c) + Ke(18,24)*(2.0*a**4 - a**2*b**2)/(b**3*c) + Ke(21,24)
     1*(2.0*a**4 - a**2*b**2)/(b**3*c) 
     1+ Ke(24,24)*(a**4 - a**2*b**2)/(b**3*c)
      Ke(2,4) = a**3*Ke(15,18)/(b*c**2) + a**3*Ke(15,21)/(b*c**2) 
     1+ a**3*Ke(18,18)/(b*c**2) + 2.0*a**3*Ke(18,21)/(b*c**2) 
     1+ a**3*Ke(18,24)/(b*c**2) + a**3*Ke(21,21)/(b*c**2) 
     1+ a**3*Ke(21,24)/(b*c**2) - a**3*Ke(15,20)/(b**2*c) 
     1- a**3*Ke(15,23)/(b**2*c) - a**3*Ke(18,23)/(b**2*c) 
     1- a**3*Ke(20,24)/(b**2*c) - a**3*Ke(21,23)/(b**2*c) 
     1- a**3*Ke(23,24)/(b**2*c) - a**2*Ke(15,16)/(b*c) 
     1- a**2*Ke(16,18)/(b*c) - a**2*Ke(16,21)/(b*c) 
     1- a**2*Ke(16,24)/(b*c) + 2.0*a**2*Ke(18,19)/(b*c) 
     1+ 2.0*a**2*Ke(18,22)/(b*c) + 2.0*a**2*Ke(19,21)/(b*c) 
     1+ 2.0*a**2*Ke(21,22)/(b*c) - 2.0*a**2*Ke(19,20)/b**2 
     1- 2.0*a**2*Ke(19,23)/b**2 - 2.0*a**2*Ke(20,22)/b**2 
     1- 2.0*a**2*Ke(22,23)/b**2 - a*Ke(11,18)/c - a*Ke(11,21)/c 
     1+ a*Ke(17,18)/c + a*Ke(17,21)/c + a*Ke(11,20)/b + a*Ke(11,23)/b 
     1- 2.0*a*Ke(16,19)/b - 2.0*a*Ke(16,22)/b - a*Ke(17,20)/b 
     1- a*Ke(17,23)/b - a*Ke(20,20)/b - a*Ke(20,23)/b + Ke(11,16) 
     1- Ke(16,17) - Ke(16,20) + (5.0*E*c*v - 4.0*E*c)/(6.0*v**2 
     1- 6.0) + Ke(18,20)*(-a**3 + a*b**2)/(b**2*c) + Ke(20,21)
     1*(-a**3 + a*b**2)/(b**2*c)
      Ke(2,5) = -a**2*Ke(11,15)/(b*c) - a**2*Ke(11,18)/(b*c) 
     1- a**2*Ke(11,21)/(b*c) - a**2*Ke(11,24)/(b*c) 
     1- a**2*Ke(15,17)/(b*c) - a**2*Ke(15,23)/(b*c) 
     1- a**2*Ke(17,18)/(b*c) - a**2*Ke(17,21)/(b*c) 
     1- a**2*Ke(17,24)/(b*c) - a**2*Ke(18,23)/(b*c) 
     1- a**2*Ke(21,23)/(b*c) - a**2*Ke(23,24)/(b*c)
     1 - 2.0*a*Ke(11,19)/b - 2.0*a*Ke(11,22)/b - 2.0*a*Ke(17,19)/b 
     1- 2.0*a*Ke(17,22)/b - 2.0*a*Ke(19,23)/b - 2.0*a*Ke(22,23)/b 
     1+ Ke(11,11) - Ke(11,20) + Ke(11,23) - Ke(17,17) - Ke(17,20) 
     1- Ke(17,23) - Ke(20,23) + (-12.0*E*a**2*c*v + 8.0*E*a**2*c 
     1+ 3.0*E*b**2*c*v - 3.0*E*b**2*c)/(12.0*a*b*v**2 - 12.0*a*b)
      Ke(2,6) = a**2*Ke(12,15)/(b*c) + a**2*Ke(12,18)/(b*c)
     1+ a**2*Ke(12,21)/(b*c) + a**2*Ke(12,24)/(b*c) + 2.0*a**2*Ke(15,23)
     1/b**2 + 2.0*a**2*Ke(18,23)/b**2 + 2.0*a**2*Ke(21,23)/b**2 
     1+ 2.0*a**2*Ke(23,24)/b**2 + 2.0*a*Ke(12,19)/b + 2.0*a*Ke(12,22)/b 
     1+ 4.0*a*c*Ke(19,23)/b**2 + 4.0*a*c*Ke(22,23)/b**2 - Ke(11,12) 
     1+ Ke(12,17) + Ke(12,20) + (-5.0*E*a*b**2*v + 8.0*E*a*b**2 
     1+ 13.0*E*a*c**2*v - 13.0*E*a*c**2)/(3.0*b**2*v**2 - 3.0*b**2) 
     1- 2.0*c*Ke(11,11)/b + 2.0*c*Ke(11,17)/b + 2.0*c*Ke(11,20)/b 
     1- 2.0*c*Ke(11,23)/b + 2.0*c*Ke(17,23)/b + 2.0*c*Ke(20,23)/b 
     1+ Ke(11,19)*(-2.0*a*b**2 + 6.0*a*c**2)/(b**2*c) + Ke(11,22)
     1*(-2.0*a*b**2 + 6.0*a*c**2)/(b**2*c) + Ke(17,19)*(2.0*a*b**2 
     1- 2.0*a*c**2)/(b**2*c) + Ke(17,22)*(2.0*a*b**2 - 2.0*a*c**2)
     1/(b**2*c) + Ke(19,20)*(2.0*a*b**2 - 2.0*a*c**2)/(b**2*c) 
     1+ Ke(20,22)*(2.0*a*b**2 - 2.0*a*c**2)/(b**2*c) + Ke(11,15)
     1*(-a**2*b**2 + 3.0*a**2*c**2)/(b**2*c**2) + Ke(11,18)*(-a**2*b**2
     1+ 3.0*a**2*c**2 + b**2*c**2)/(b**2*c**2) + Ke(11,21)*(-a**2*b**2 
     1+ 3.0*a**2*c**2)/(b**2*c**2) + Ke(11,24)*(-a**2*b**2 
     1+ 3.0*a**2*c**2 - b**2*c**2)/(b**2*c**2) + Ke(15,17)*(a**2*b**2 
     1- a**2*c**2)/(b**2*c**2) + Ke(15,20)*(a**2*b**2 - a**2*c**2)
     1/(b**2*c**2) + Ke(17,18)*(a**2*b**2 - a**2*c**2 - b**2*c**2)
     1/(b**2*c**2) + Ke(17,21)*(a**2*b**2 - a**2*c**2)/(b**2*c**2) 
     1+ Ke(17,24)*(a**2*b**2 - a**2*c**2 + b**2*c**2)/(b**2*c**2) 
     1+ Ke(18,20)*(a**2*b**2 - a**2*c**2 - b**2*c**2)/(b**2*c**2) 
     1+ Ke(20,21)*(a**2*b**2 - a**2*c**2)/(b**2*c**2) + Ke(20,24)
     1*(a**2*b**2 - a**2*c**2 + b**2*c**2)/(b**2*c**2) + Ke(19,19)
     1*(4.0*a**2*b**2 - 4.0*a**2*c**2)/(b**3*c) + Ke(19,22)
     1*(8.0*a**2*b**2 - 8.0*a**2*c**2)/(b**3*c) + Ke(22,22)
     1*(4.0*a**2*b**2 - 4.0*a**2*c**2)/(b**3*c) + Ke(15,19)
     1*(4.0*a**3*b**2 - 4.0*a**3*c**2)/(b**3*c**2) + Ke(15,22)
     1*(4.0*a**3*b**2 - 4.0*a**3*c**2)/(b**3*c**2) + Ke(18,19)
     1*(4.0*a**3*b**2 - 4.0*a**3*c**2 - 2.0*a*b**2*c**2)/(b**3*c**2)
     1+ Ke(18,22)*(4.0*a**3*b**2 - 4.0*a**3*c**2 - 2.0*a*b**2*c**2)
     1/(b**3*c**2) + Ke(19,21)*(4.0*a**3*b**2 - 4.0*a**3*c**2)
     1/(b**3*c**2) + Ke(19,24)*(4.0*a**3*b**2 - 4.0*a**3*c**2 
     1+ 2.0*a*b**2*c**2)/(b**3*c**2) + Ke(21,22)*(4.0*a**3*b**2 
     1- 4.0*a**3*c**2)/(b**3*c**2) + Ke(22,24)*(4.0*a**3*b**2 
     1- 4.0*a**3*c**2 + 2.0*a*b**2*c**2)/(b**3*c**2) + Ke(15,15)
     1*(a**4*b**2 - a**4*c**2)/(b**3*c**3) + Ke(15,18)*(2.0*a**4*b**2 
     1- 2.0*a**4*c**2 - a**2*b**2*c**2)/(b**3*c**3) + Ke(15,21)
     1*(2.0*a**4*b**2 - 2.0*a**4*c**2)/(b**3*c**3) + Ke(15,24)
     1*(2.0*a**4*b**2 - 2.0*a**4*c**2 + a**2*b**2*c**2)/(b**3*c**3) 
     1+ Ke(18,18)*(a**4*b**2 - a**4*c**2 - a**2*b**2*c**2)/(b**3*c**3) 
     1+ Ke(18,21)*(2.0*a**4*b**2 - 2.0*a**4*c**2 - a**2*b**2*c**2)
     1/(b**3*c**3) + Ke(18,24)*(2.0*a**4*b**2-2.0*a**4*c**2)/(b**3*c**3)
     1+ Ke(21,21)*(a**4*b**2 - a**4*c**2)/(b**3*c**3) 
     1+ Ke(21,24)*(2.0*a**4*b**2 - 2.0*a**4*c**2 + a**2*b**2*c**2)
     1/(b**3*c**3) + Ke(24,24)*(a**4*b**2 - a**4*c**2 
     1+ a**2*b**2*c**2)/(b**3*c**3)
      Ke(2,7)= -E*c/(12.0*v - 12.0) - a**3*Ke(15,18)/(b*c**2) 
     1- a**3*Ke(15,21)/(b*c**2) - a**3*Ke(18,18)/(b*c**2) 
     1- 2.0*a**3*Ke(18,21)/(b*c**2) - a**3*Ke(18,24)/(b*c**2) 
     1- a**3*Ke(21,21)/(b*c**2) - a**3*Ke(21,24)/(b*c**2) 
     1+ a**3*Ke(15,20)/(b**2*c) + a**3*Ke(15,23)/(b**2*c) 
     1+ a**3*Ke(18,23)/(b**2*c) + a**3*Ke(20,24)/(b**2*c) 
     1+ a**3*Ke(21,23)/(b**2*c) + a**3*Ke(23,24)/(b**2*c) 
     1- a**2*Ke(15,19)/(b*c) - 3.0*a**2*Ke(18,19)/(b*c) 
     1- 2.0*a**2*Ke(18,22)/(b*c) - 3.0*a**2*Ke(19,21)/(b*c) 
     1- a**2*Ke(19,24)/(b*c) - 2.0*a**2*Ke(21,22)/(b*c) 
     1+ 2.0*a**2*Ke(19,23)/b**2 + 2.0*a**2*Ke(20,22)/b**2 
     1+ 2.0*a**2*Ke(22,23)/b**2 + a*Ke(11,18)/c + a*Ke(11,21)/c 
     1- a*Ke(17,18)/c - a*Ke(17,21)/c - a*Ke(11,20)/b - a*Ke(11,23)/b 
     1+ a*Ke(17,20)/b + a*Ke(17,23)/b - 2.0*a*Ke(19,19)/b 
     1- 2.0*a*Ke(19,22)/b + a*Ke(20,20)/b + a*Ke(20,23)/b + Ke(11,19) 
     1- Ke(17,19) + Ke(19,20)*(2.0*a**2 - b**2)/b**2 + Ke(18,20)*(a**3 
     1- a*b**2)/(b**2*c) + Ke(20,21)*(a**3 - a*b**2)/(b**2*c)
      Ke(2,8) = -a**4*Ke(15,15)/(b**2*c**2) - 2.0*a**4*Ke(15,18)
     1/(b**2*c**2) - 2.0*a**4*Ke(15,21)/(b**2*c**2) - 2.0*a**4*Ke(15,24)
     1/(b**2*c**2) - a**4*Ke(18,18)/(b**2*c**2) - 2.0*a**4*Ke(18,21)
     1/(b**2*c**2) - 2.0*a**4*Ke(18,24)/(b**2*c**2) - a**4*Ke(21,21)
     1/(b**2*c**2) - 2.0*a**4*Ke(21,24)/(b**2*c**2) - a**4*Ke(24,24)
     1/(b**2*c**2) - 4.0*a**3*Ke(15,19)/(b**2*c) - 4.0*a**3*Ke(15,22)
     1/(b**2*c) - 4.0*a**3*Ke(18,19)/(b**2*c) - 4.0*a**3*Ke(18,22)
     1/(b**2*c) - 4.0*a**3*Ke(19,21)/(b**2*c) - 4.0*a**3*Ke(19,24)
     1/(b**2*c) - 4.0*a**3*Ke(21,22)/(b**2*c) - 4.0*a**3*Ke(22,24)
     1/(b**2*c) + 2.0*a**2*Ke(11,15)/(b*c) + 2.0*a**2*Ke(11,18)/(b*c) 
     1+ 2.0*a**2*Ke(11,21)/(b*c) + 2.0*a**2*Ke(11,24)/(b*c) 
     1- a**2*Ke(15,17)/(b*c) - 2.0*a**2*Ke(15,20)/(b*c) + a**2*Ke(15,23)
     1/(b*c) - a**2*Ke(17,18)/(b*c) - a**2*Ke(17,21)/(b*c) 
     1- a**2*Ke(17,24)/(b*c) - 2.0*a**2*Ke(18,20)/(b*c) + a**2*Ke(18,23)
     1/(b*c) - 2.0*a**2*Ke(20,21)/(b*c) - 2.0*a**2*Ke(20,24)/(b*c) 
     1+ a**2*Ke(21,23)/(b*c) + a**2*Ke(23,24)/(b*c) - 4.0*a**2*Ke(19,19)
     1/b**2 - 8.0*a**2*Ke(19,22)/b**2 - 4.0*a**2*Ke(22,22)/b**2 
     1+ 4.0*a*Ke(11,19)/b + 4.0*a*Ke(11,22)/b - 2.0*a*Ke(17,19)/b 
     1- 2.0*a*Ke(17,22)/b - 4.0*a*Ke(19,20)/b + 2.0*a*Ke(19,23)/b 
     1- 4.0*a*Ke(20,22)/b + 2.0*a*Ke(22,23)/b - Ke(11,11) + Ke(11,17) 
     1+ 2.0*Ke(11,20) - Ke(11,23) - Ke(17,20) + Ke(17,23) - Ke(20,20) 
     1+ Ke(20,23) + (44.0*E*a**2*c*v - 46.0*E*a**2*c + 3.0*E*b**2*c*v 
     1- 3.0*E*b**2*c)/(12.0*a*b*v**2 - 12.0*a*b)
      Ke(2,9) = -a**4*Ke(15,15)/(b*c**3) - 2.0*a**4*Ke(15,18)/(b*c**3) 
     1-a**4*Ke(18,18)/(b*c**3) - 4.0*a**3*Ke(15,19)/(b*c**2) 
     1- 4.0*a**3*Ke(15,22)/(b*c**2) - 4.0*a**3*Ke(18,19)/(b*c**2) 
     1- 4.0*a**3*Ke(18,22)/(b*c**2) + a**2*Ke(11,15)/c**2 
     1+ a**2*Ke(11,18)/c**2 - a**2*Ke(15,17)/c**2 - a**2*Ke(15,20)/c**2 
     1- a**2*Ke(17,18)/c**2 - a**2*Ke(18,20)/c**2 - a**2*Ke(12,15)/(b*c)
     1- a**2*Ke(12,18)/(b*c) - a**2*Ke(12,21)/(b*c)-a**2*Ke(12,24)/(b*c)
     1- 4.0*a**2*Ke(19,19)/(b*c) - 8.0*a**2*Ke(19,22)/(b*c) 
     1- 4.0*a**2*Ke(22,22)/(b*c) + 2.0*a*Ke(11,19)/c + 2.0*a*Ke(11,22)/c
     1- 2.0*a*Ke(17,19)/c - 2.0*a*Ke(17,22)/c - 2.0*a*Ke(19,20)/c 
     1- 2.0*a*Ke(20,22)/c - 2.0*a*Ke(12,19)/b - 2.0*a*Ke(12,22)/b 
     1+ Ke(11,12) - Ke(12,17) - Ke(12,20) + (5.0*E*a*v - 8.0*E*a)
     1/(3.0*v**2 - 3.0) + Ke(11,21)*(a**2 + c**2)/c**2 + Ke(11,24)
     1*(a**2 + c**2)/c**2 + Ke(17,21)*(-a**2 - c**2)/c**2 + Ke(17,24)
     1*(-a**2 - c**2)/c**2 + Ke(20,21)*(-a**2 - c**2)/c**2 + Ke(20,24)
     1*(-a**2 - c**2)/c**2 + Ke(19,21)*(-4.0*a**3 - 2.0*a*c**2)/(b*c**2)
     1+ Ke(19,24)*(-4.0*a**3 - 2.0*a*c**2)/(b*c**2) + Ke(21,22)
     1*(-4.0*a**3 - 2.0*a*c**2)/(b*c**2) + Ke(22,24)*(-4.0*a**3 
     1- 2.0*a*c**2)/(b*c**2) + Ke(15,21)*(-2.0*a**4 -a**2*c**2)/(b*c**3)
     1+ Ke(15,24)*(-2.0*a**4 - a**2*c**2)/(b*c**3) + Ke(18,21)
     1*(-2.0*a**4 - a**2*c**2)/(b*c**3) + Ke(18,24)*(-2.0*a**4 
     1- a**2*c**2)/(b*c**3) + Ke(21,21)*(-a**4 - a**2*c**2)/(b*c**3) 
     1+ Ke(21,24)*(-2.0*a**4 - 2.0*a**2*c**2)/(b*c**3) 
     1+ Ke(24,24)*(-a**4 - a**2*c**2)/(b*c**3)
      Ke(2,10) = -a**3*Ke(15,15)/(b*c**2) - a**3*Ke(15,18)/(b*c**2) 
     1- a**3*Ke(15,21)/(b*c**2) - 2.0*a**3*Ke(15,24)/(b*c**2) 
     1- a**3*Ke(18,24)/(b*c**2) - a**3*Ke(21,24)/(b*c**2) 
     1- a**3*Ke(24,24)/(b*c**2) - a**3*Ke(15,23)/(b**2*c) 
     1- a**3*Ke(18,20)/(b**2*c) - a**3*Ke(18,23)/(b**2*c) 
     1- a**3*Ke(20,21)/(b**2*c) - a**3*Ke(21,23)/(b**2*c) 
     1- a**3*Ke(23,24)/(b**2*c) - 2.0*a**2*Ke(15,19)/(b*c) 
     1- 3.0*a**2*Ke(15,22)/(b*c) - a**2*Ke(18,22)/(b*c) 
     1- 2.0*a**2*Ke(19,24)/(b*c) - a**2*Ke(21,22)/(b*c) 
     1- 3.0*a**2*Ke(22,24)/(b*c) - 2.0*a**2*Ke(19,20)/b**2 
     1- 2.0*a**2*Ke(19,23)/b**2 - 2.0*a**2*Ke(22,23)/b**2 
     1+ a*Ke(11,15)/c + a*Ke(11,24)/c - a*Ke(15,17)/c - a*Ke(17,24)/c 
     1+ a*Ke(11,20)/b + a*Ke(11,23)/b - a*Ke(17,20)/b - a*Ke(17,23)/b 
     1- 2.0*a*Ke(19,22)/b - a*Ke(20,20)/b - a*Ke(20,23)/b 
     1- 2.0*a*Ke(22,22)/b + Ke(11,22) - Ke(17,22) + (13.0*E*c*v 
     1- 15.0*E*c)/(12.0*v**2 - 12.0) + Ke(20,22)*(-2.0*a**2 - b**2)/b**2
     1+ Ke(15,20)*(-a**3 - a*b**2)/(b**2*c) + Ke(20,24)*(-a**3 
     1- a*b**2)/(b**2*c)
      Ke(2,11) = a**2*Ke(11,15)/(b*c) + a**2*Ke(11,18)/(b*c) 
     1+ a**2*Ke(11,21)/(b*c) + a**2*Ke(11,24)/(b*c) + 2.0*a*Ke(11,19)/b 
     1+ 2.0*a*Ke(11,22)/b - Ke(11,11) + Ke(11,17) + Ke(11,20) 
     1+ (8.0*E*a**2*c*v - 6.0*E*a**2*c + 3.0*E*b**2*c*v 
     1- 3.0*E*b**2*c)/(12.0*a*b*v**2 - 12.0*a*b)
      Ke(2,12) = a**2*Ke(12,15)/(b*c) + a**2*Ke(12,18)/(b*c) 
     1+ a**2*Ke(12,21)/(b*c) + a**2*Ke(12,24)/(b*c) + 2.0*a*Ke(12,19)/b 
     1+ 2.0*a*Ke(12,22)/b - Ke(11,12) + Ke(12,17) + Ke(12,20)
      Ke(2,13) = a**3*Ke(15,15)/(b*c**2) + 2.0*a**3*Ke(15,18)/(b*c**2) 
     1+ 2.0*a**3*Ke(15,21)/(b*c**2) + 2.0*a**3*Ke(15,24)/(b*c**2) 
     1+ a**3*Ke(18,18)/(b*c**2) + 2.0*a**3*Ke(18,21)/(b*c**2) 
     1+ 2.0*a**3*Ke(18,24)/(b*c**2) + a**3*Ke(21,21)/(b*c**2) 
     1+ 2.0*a**3*Ke(21,24)/(b*c**2) + a**3*Ke(24,24)/(b*c**2)
     1- a**2*Ke(15,16)/(b*c) + 3.0*a**2*Ke(15,19)/(b*c) 
     1+ 3.0*a**2*Ke(15,22)/(b*c) - a**2*Ke(16,18)/(b*c) 
     1- a**2*Ke(16,21)/(b*c) - a**2*Ke(16,24)/(b*c) + 3.0*a**2*Ke(18,19)
     1/(b*c) + 3.0*a**2*Ke(18,22)/(b*c) + 3.0*a**2*Ke(19,21)/(b*c) 
     1+ 3.0*a**2*Ke(19,24)/(b*c) + 3.0*a**2*Ke(21,22)/(b*c) 
     1+ 3.0*a**2*Ke(22,24)/(b*c) - a*Ke(11,15)/c - a*Ke(11,18)/c 
     1- a*Ke(11,21)/c - a*Ke(11,24)/c + a*Ke(15,17)/c + a*Ke(15,20)/c 
     1+ a*Ke(17,18)/c + a*Ke(17,21)/c + a*Ke(17,24)/c + a*Ke(18,20)/c 
     1+ a*Ke(20,21)/c + a*Ke(20,24)/c - 2.0*a*Ke(16,19)/b 
     1- 2.0*a*Ke(16,22)/b + 2.0*a*Ke(19,19)/b + 4.0*a*Ke(19,22)/b 
     1+ 2.0*a*Ke(22,22)/b + Ke(11,16) - Ke(11,19) - Ke(11,22) 
     1- Ke(16,17) - Ke(16,20) + Ke(17,19) + Ke(17,22) + Ke(19,20) 
     1+ Ke(20,22) + (-11.0*E*c*v + 15.0*E*c)/(12.0*v**2 - 12.0)
      Ke(2,14) = -a**2*Ke(15,17)/(b*c) - a**2*Ke(15,20)/(b*c) 
     1- a**2*Ke(15,23)/(b*c) - a**2*Ke(17,18)/(b*c)-a**2*Ke(17,21)/(b*c)
     1- a**2*Ke(17,24)/(b*c) - a**2*Ke(18,20)/(b*c)-a**2*Ke(18,23)/(b*c)
     1- a**2*Ke(20,21)/(b*c) - a**2*Ke(20,24)/(b*c)-a**2*Ke(21,23)/(b*c)
     1- a**2*Ke(23,24)/(b*c) - 2.0*a*Ke(17,19)/b - 2.0*a*Ke(17,22)/b 
     1- 2.0*a*Ke(19,20)/b - 2.0*a*Ke(19,23)/b - 2.0*a*Ke(20,22)/b 
     1- 2.0*a*Ke(22,23)/b + Ke(11,17) + Ke(11,20) + Ke(11,23) 
     1- Ke(17,17) - 2.0*Ke(17,20) - Ke(17,23) - Ke(20,20) - Ke(20,23) 
     1+ (2.0*E*a**2*c*v - 3.0*E*a**2*c + 3.0*E*b**2*c*v 
     1- 3.0*E*b**2*c)/(6.0*a*b*v**2 - 6.0*a*b)
      Ke(2,15) = a**2*Ke(15,15)/(b*c) + a**2*Ke(15,18)/(b*c) 
     1+ a**2*Ke(15,21)/(b*c) + a**2*Ke(15,24)/(b*c) + 2.0*a*Ke(15,19)/b 
     1+ 2.0*a*Ke(15,22)/b - Ke(11,15) + Ke(15,17) + Ke(15,20)
      Ke(2,16) = a**2*Ke(15,16)/(b*c) + a**2*Ke(16,18)/(b*c) 
     1+ a**2*Ke(16,21)/(b*c) + a**2*Ke(16,24)/(b*c) + 2.0*a*Ke(16,19)/b 
     1+ 2.0*a*Ke(16,22)/b - Ke(11,16) + Ke(16,17) + Ke(16,20) 
     1+ (-3.0*E*c*v + E*c)/(12.0*v**2 - 12.0)
      Ke(2,17) = a**2*Ke(15,17)/(b*c) + a**2*Ke(17,18)/(b*c) 
     1+ a**2*Ke(17,21)/(b*c) + a**2*Ke(17,24)/(b*c) + 2.0*a*Ke(17,19)/b 
     1+ 2.0*a*Ke(17,22)/b - Ke(11,17) + Ke(17,17) + Ke(17,20) 
     1+ (-4.0*E*a**2*c*v + 6.0*E*a**2*c - 3.0*E*b**2*c*v 
     1+ 3.0*E*b**2*c)/(12.0*a*b*v**2 - 12.0*a*b)
      Ke(2,18) = a**2*Ke(15,18)/(b*c) + a**2*Ke(18,18)/(b*c) 
     1+ a**2*Ke(18,21)/(b*c) + a**2*Ke(18,24)/(b*c) + 2.0*a*Ke(18,19)/b 
     1+ 2.0*a*Ke(18,22)/b - Ke(11,18) + Ke(17,18) + Ke(18,20)
      Ke(2,19) = a**2*Ke(15,19)/(b*c) + a**2*Ke(18,19)/(b*c) 
     1+ a**2*Ke(19,21)/(b*c) + a**2*Ke(19,24)/(b*c) + 2.0*a*Ke(19,19)/b 
     1+ 2.0*a*Ke(19,22)/b - Ke(11,19) + Ke(17,19) + Ke(19,20) 
     1+ (-3.0*E*c*v + 5.0*E*c)/(12.0*v**2 - 12.0)
      Ke(2,20) = a**2*Ke(15,20)/(b*c) + a**2*Ke(18,20)/(b*c) 
     1+ a**2*Ke(20,21)/(b*c) + a**2*Ke(20,24)/(b*c) + 2.0*a*Ke(19,20)/b 
     1+ 2.0*a*Ke(20,22)/b - Ke(11,20) + Ke(17,20) + Ke(20,20) 
     1+ (-8.0*E*a**2*c*v + 6.0*E*a**2*c - 3.0*E*b**2*c*v 
     1+ 3.0*E*b**2*c)/(12.0*a*b*v**2 - 12.0*a*b)
      Ke(2,21) = a**2*Ke(15,21)/(b*c) + a**2*Ke(18,21)/(b*c) 
     1+ a**2*Ke(21,21)/(b*c) + a**2*Ke(21,24)/(b*c) + 2.0*a*Ke(19,21)/b 
     1+ 2.0*a*Ke(21,22)/b - Ke(11,21) + Ke(17,21) + Ke(20,21)
      Ke(2,22) = a**2*Ke(15,22)/(b*c) + a**2*Ke(18,22)/(b*c) 
     1+ a**2*Ke(21,22)/(b*c) + a**2*Ke(22,24)/(b*c) + 2.0*a*Ke(19,22)/b 
     1+ 2.0*a*Ke(22,22)/b - Ke(11,22) + Ke(17,22) + Ke(20,22) 
     1+ (-9.0*E*c*v + 11.0*E*c)/(12.0*v**2 - 12.0)
      Ke(2,23) = a**2*Ke(15,23)/(b*c) + a**2*Ke(18,23)/(b*c) 
     1+ a**2*Ke(21,23)/(b*c) + a**2*Ke(23,24)/(b*c) + 2.0*a*Ke(19,23)/b 
     1+ 2.0*a*Ke(22,23)/b - Ke(11,23) + Ke(17,23) + Ke(20,23) 
     1+ (8.0*E*a**2*c*v - 6.0*E*a**2*c - 3.0*E*b**2*c*v 
     1+ 3.0*E*b**2*c)/(12.0*a*b*v**2 - 12.0*a*b)
      Ke(2,24) = a**2*Ke(15,24)/(b*c) + a**2*Ke(18,24)/(b*c) 
     1+ a**2*Ke(21,24)/(b*c) + a**2*Ke(24,24)/(b*c) + 2.0*a*Ke(19,24)/b 
     1+ 2.0*a*Ke(22,24)/b - Ke(11,24) + Ke(17,24) + Ke(20,24)
      Ke(3,3) =-16.0*E*a*c**3/(3.0*b**3*v + 3.0*b**3) 
     1+ a**4*Ke(18,18)/b**4 + 2.0*a**4*Ke(18,21)/b**4 
     1+ a**4*Ke(21,21)/b**4 + 4.0*a**3*c*Ke(18,19)/b**4 
     1+ 4.0*a**3*c*Ke(18,22)/b**4 + 4.0*a**3*c*Ke(19,21)/b**4 
     1+ 4.0*a**3*c*Ke(21,22)/b**4 - 2.0*a**2*Ke(12,18)/b**2 
     1- 2.0*a**2*Ke(12,21)/b**2 - 4.0*a**2*c*Ke(11,18)/b**3 
     1- 4.0*a**2*c*Ke(11,21)/b**3 - 4.0*a**2*c*Ke(18,23)/b**3 
     1- 4.0*a**2*c*Ke(21,23)/b**3 + 4.0*a**2*c**2*Ke(19,19)/b**4 
     1+ 8.0*a**2*c**2*Ke(19,22)/b**4 + 4.0*a**2*c**2*Ke(22,22)/b**4 
     1- 4.0*a*c*Ke(12,19)/b**2 - 4.0*a*c*Ke(12,22)/b**2 
     1- 8.0*a*c**2*Ke(11,19)/b**3 - 8.0*a*c**2*Ke(11,22)/b**3 
     1- 8.0*a*c**2*Ke(19,23)/b**3 - 8.0*a*c**2*Ke(22,23)/b**3 +Ke(12,12)
     1+ 4.0*c*Ke(11,12)/b + 4.0*c*Ke(12,23)/b + 4.0*c**2*Ke(11,11)/b**2
     1+ 8.0*c**2*Ke(11,23)/b**2 + 4.0*c**2*Ke(23,23)/b**2 + Ke(12,15)
     1*(-2.0*a**2 + 2.0*b**2)/b**2 + Ke(12,24)*(-2.0*a**2 + 2.0*b**2)
     1/b**2 + Ke(11,15)*(-4.0*a**2*c + 4.0*b**2*c)/b**3 + Ke(11,24)
     1*(-4.0*a**2*c + 4.0*b**2*c)/b**3 + Ke(15,23)*(-4.0*a**2*c 
     1+ 4.0*b**2*c)/b**3 + Ke(23,24)*(-4.0*a**2*c + 4.0*b**2*c)/b**3 
     1+ Ke(15,15)*(a**4 - 2.0*a**2*b**2 + b**4)/b**4 + Ke(15,18)
     1*(2.0*a**4 - 2.0*a**2*b**2)/b**4 + Ke(15,19)*(4.0*a**3*c 
     1- 4.0*a*b**2*c)/b**4 + Ke(15,21)*(2.0*a**4 - 2.0*a**2*b**2)
     1/b**4 + Ke(15,22)*(4.0*a**3*c - 4.0*a*b**2*c)/b**4 + Ke(15,24)
     1*(2.0*a**4 - 4.0*a**2*b**2 + 2.0*b**4)/b**4 + Ke(18,24)*(2.0*a**4
     1- 2.0*a**2*b**2)/b**4 + Ke(19,24)*(4.0*a**3*c - 4.0*a*b**2*c)
     1/b**4 + Ke(21,24)*(2.0*a**4 - 2.0*a**2*b**2)/b**4 + Ke(22,24)
     1*(4.0*a**3*c - 4.0*a*b**2*c)/b**4 
     1+ Ke(24,24)*(a**4 - 2.0*a**2*b**2 + b**4)/b**4
      Ke(3,4) = E*c**2/(3.0*b*v + 3.0*b) + a**3*Ke(18,18)/(b**2*c) 
     1 + 2.0*a**3*Ke(18,21)/(b**2*c) + a**3*Ke(21,21)/(b**2*c) 
     1 - a**3*Ke(18,20)/b**3 - a**3*Ke(20,21)/b**3 - a**2*Ke(16,18)
     1 /b**2 - a**2*Ke(16,21)/b**2 + 2.0*a**2*Ke(18,19)/b**2 
     1+ 2.0*a**2*Ke(18,22)/b**2 + 2.0*a**2*Ke(19,21)/b**2 
     1+ 2.0*a**2*Ke(21,22)/b**2 - 2.0*a**2*c*Ke(19,20)/b**3 
     1- 2.0*a**2*c*Ke(19,23)/b**3 - 2.0*a**2*c*Ke(20,22)/b**3 
     1- 2.0*a**2*c*Ke(22,23)/b**3 - a*Ke(12,18)/c - a*Ke(12,21)/c 
     1- 2.0*a*Ke(11,18)/b - 2.0*a*Ke(11,21)/b + a*Ke(12,20)/b 
     1+ a*Ke(12,23)/b + 2.0*a*c*Ke(11,20)/b**2 + 2.0*a*c*Ke(11,23)/b**2 
     1- 2.0*a*c*Ke(16,19)/b**2 - 2.0*a*c*Ke(16,22)/b**2 
     1+ 2.0*a*c*Ke(20,23)/b**2 + 2.0*a*c*Ke(23,23)/b**2 + Ke(12,16) 
     1+ 2.0*c*Ke(11,16)/b + 2.0*c*Ke(16,23)/b + Ke(15,16)*(-a**2 
     1+ b**2)/b**2 + Ke(16,24)*(-a**2 + b**2)/b**2 + Ke(15,18)*(a**3 
     1- a*b**2)/(b**2*c) + Ke(15,21)*(a**3 - a*b**2)/(b**2*c) 
     1+ Ke(18,24)*(a**3 - a*b**2)/(b**2*c) + Ke(21,24)*(a**3 - a*b**2)
     1/(b**2*c) + Ke(15,20)*(-a**3 + a*b**2)/b**3 + Ke(15,23)*(-a**3 
     1+ a*b**2)/b**3 + Ke(18,23)*(-a**3 - 2.0*a*b**2)/b**3 + Ke(20,24)
     1*(-a**3 + a*b**2)/b**3 + Ke(21,23)*(-a**3 - 2.0*a*b**2)/b**3 
     1+ Ke(23,24)*(-a**3 + a*b**2)/b**3
      Ke(3,5) = -E*a*c**2/(b**2*v + b**2) - a**2*Ke(11,18)/b**2 
     1- a**2*Ke(11,21)/b**2 - a**2*Ke(17,18)/b**2 - a**2*Ke(17,21)
     1/b**2 - a**2*Ke(18,23)/b**2 - a**2*Ke(21,23)/b**2 
     1- 2.0*a*c*Ke(11,19)/b**2 - 2.0*a*c*Ke(11,22)/b**2 
     1- 2.0*a*c*Ke(17,19)/b**2 - 2.0*a*c*Ke(17,22)/b**2 
     1- 2.0*a*c*Ke(19,23)/b**2 - 2.0*a*c*Ke(22,23)/b**2 
     1+ Ke(11,12) + Ke(12,17) + Ke(12,23) + 2.0*c*Ke(11,11)/b 
     1+ 2.0*c*Ke(11,17)/b + 4.0*c*Ke(11,23)/b + 2.0*c*Ke(17,23)/b 
     1+ 2.0*c*Ke(23,23)/b + Ke(11,15)*(-a**2 + b**2)/b**2 
     1+ Ke(11,24)*(-a**2 + b**2)/b**2 + Ke(15,17)*(-a**2 + b**2)/b**2 
     1+ Ke(15,23)*(-a**2 + b**2)/b**2 + Ke(17,24)*(-a**2 + b**2)/b**2 
     1+ Ke(23,24)*(-a**2 + b**2)/b**2
      Ke(3,6) = -Ke(12,12) + (-8.0*E*a*b**2*c + 16.0*E*a*c**3)
     1/(3.0*b**3*v + 3.0*b**3) - 4.0*c*Ke(11,12)/b - 4.0*c*Ke(12,23)/b 
     1- 4.0*c**2*Ke(11,11)/b**2 - 8.0*c**2*Ke(11,23)/b**2 
     1- 4.0*c**2*Ke(23,23)/b**2 + Ke(12,19)*(-2.0*a*b**2 + 4.0*a*c**2)
     1/(b**2*c) + Ke(12,22)*(-2.0*a*b**2 + 4.0*a*c**2)/(b**2*c) 
     1+ Ke(12,15)*(-a**2*b**2 + 2.0*a**2*c**2 - b**2*c**2)/(b**2*c**2) 
     1+ Ke(12,18)*(-a**2*b**2 + 2.0*a**2*c**2 + b**2*c**2)/(b**2*c**2) 
     1+ Ke(12,21)*(-a**2*b**2 + 2.0*a**2*c**2)/(b**2*c**2) + Ke(12,24)
     1*(-a**2*b**2 + 2.0*a**2*c**2 - 2.0*b**2*c**2)/(b**2*c**2) 
     1+ Ke(11,19)*(-4.0*a*b**2 + 8.0*a*c**2)/b**3 + Ke(11,22)
     1*(-4.0*a*b**2 + 8.0*a*c**2)/b**3 + Ke(19,23)*(-4.0*a*b**2 
     1+ 8.0*a*c**2)/b**3 + Ke(22,23)*(-4.0*a*b**2 + 8.0*a*c**2)/b**3 
     1+ Ke(11,15)*(-2.0*a**2*b**2 + 4.0*a**2*c**2 - 2.0*b**2*c**2)
     1/(b**3*c) + Ke(11,18)*(-2.0*a**2*b**2 + 4.0*a**2*c**2 
     1+ 2.0*b**2*c**2)/(b**3*c) + Ke(11,21)*(-2.0*a**2*b**2 
     1+ 4.0*a**2*c**2)/(b**3*c) + Ke(11,24)*(-2.0*a**2*b**2 
     1+ 4.0*a**2*c**2 - 4.0*b**2*c**2)/(b**3*c) + Ke(15,23)
     1*(-2.0*a**2*b**2 + 4.0*a**2*c**2 - 2.0*b**2*c**2)/(b**3*c) 
     1+ Ke(18,23)*(-2.0*a**2*b**2 + 4.0*a**2*c**2 + 2.0*b**2*c**2)
     1/(b**3*c) + Ke(21,23)*(-2.0*a**2*b**2 + 4.0*a**2*c**2)/(b**3*c) 
     1+ Ke(23,24)*(-2.0*a**2*b**2 + 4.0*a**2*c**2 - 4.0*b**2*c**2)
     1/(b**3*c) + Ke(19,19)*(4.0*a**2*b**2 - 4.0*a**2*c**2)/b**4
     1+ Ke(19,22)*(8.0*a**2*b**2 - 8.0*a**2*c**2)/b**4 + Ke(22,22)
     1*(4.0*a**2*b**2 - 4.0*a**2*c**2)/b**4 + Ke(15,19)*(4.0*a**3*b**2 
     1- 4.0*a**3*c**2 - 2.0*a*b**4 + 2.0*a*b**2*c**2)/(b**4*c) 
     1+ Ke(15,22)*(4.0*a**3*b**2 - 4.0*a**3*c**2 - 2.0*a*b**4 
     1+ 2.0*a*b**2*c**2)/(b**4*c) + Ke(18,19)*(4.0*a**3*b**2 
     1- 4.0*a**3*c**2 - 2.0*a*b**2*c**2)/(b**4*c) + Ke(18,22)
     1*(4.0*a**3*b**2 - 4.0*a**3*c**2 - 2.0*a*b**2*c**2)/(b**4*c) 
     1+ Ke(19,21)*(4.0*a**3*b**2 - 4.0*a**3*c**2)/(b**4*c) 
     1+ Ke(19,24)*(4.0*a**3*b**2 - 4.0*a**3*c**2 - 2.0*a*b**4 
     1+ 4.0*a*b**2*c**2)/(b**4*c) + Ke(21,22)*(4.0*a**3*b**2 
     1- 4.0*a**3*c**2)/(b**4*c) + Ke(22,24)*(4.0*a**3*b**2 
     1- 4.0*a**3*c**2 - 2.0*a*b**4 + 4.0*a*b**2*c**2)/(b**4*c) 
     1+ Ke(15,15)*(a**4*b**2 - a**4*c**2 - a**2*b**4 + a**2*b**2*c**2)
     1/(b**4*c**2) + Ke(15,18)*(2.0*a**4*b**2 - 2.0*a**4*c**2 
     1- a**2*b**4 + b**4*c**2)/(b**4*c**2) + Ke(15,21)*(2.0*a**4*b**2 
     1- 2.0*a**4*c**2 - a**2*b**4 + a**2*b**2*c**2)/(b**4*c**2) 
     1+ Ke(15,24)*(2.0*a**4*b**2 - 2.0*a**4*c**2 - 2.0*a**2*b**4 
     1+ 3.0*a**2*b**2*c**2 - b**4*c**2)/(b**4*c**2) + Ke(18,18)
     1*(a**4*b**2 - a**4*c**2 - a**2*b**2*c**2)/(b**4*c**2) 
     1+ Ke(18,21)*(2.0*a**4*b**2 - 2.0*a**4*c**2 - a**2*b**2*c**2)
     1/(b**4*c**2) + Ke(18,24)*(2.0*a**4*b**2 - 2.0*a**4*c**2 
     1- a**2*b**4 + a**2*b**2*c**2 + b**4*c**2)/(b**4*c**2) + Ke(21,21)
     1 *(a**4*b**2 - a**4*c**2)/(b**4*c**2) + Ke(21,24)*(2.0*a**4*b**2 
     1- 2.0*a**4*c**2 - a**2*b**4 + 2.0*a**2*b**2*c**2)/(b**4*c**2) 
     1+ Ke(24,24)*(a**4*b**2 - a**4*c**2 - a**2*b**4 
     1+ 2.0*a**2*b**2*c**2 - b**4*c**2)/(b**4*c**2)
      Ke(3,7) = 2.0*E*c**2/(3.0*b*v + 3.0*b) - a**3*Ke(18,18)/(b**2*c) 
     1- 2.0*a**3*Ke(18,21)/(b**2*c) - a**3*Ke(21,21)/(b**2*c) 
     1+ a**3*Ke(18,20)/b**3 + a**3*Ke(20,21)/b**3 - 3.0*a**2*Ke(18,19)
     1/b**2 - 2.0*a**2*Ke(18,22)/b**2 - 3.0*a**2*Ke(19,21)/b**2 
     1- 2.0*a**2*Ke(21,22)/b**2 + 2.0*a**2*c*Ke(19,20)/b**3 
     1+ 2.0*a**2*c*Ke(20,22)/b**3 + 2.0*a**2*c*Ke(22,23)/b**3 
     1+ a*Ke(12,18)/c + a*Ke(12,21)/c + 2.0*a*Ke(11,18)/b 
     1+ 2.0*a*Ke(11,21)/b - a*Ke(12,20)/b - a*Ke(12,23)/b 
     1- 2.0*a*c*Ke(11,20)/b**2 - 2.0*a*c*Ke(11,23)/b**2 
     1- 2.0*a*c*Ke(19,19)/b**2 - 2.0*a*c*Ke(19,22)/b**2 
     1- 2.0*a*c*Ke(20,23)/b**2 - 2.0*a*c*Ke(23,23)/b**2 
     1+ Ke(12,19) + 2.0*c*Ke(11,19)/b + Ke(15,19)*(-a**2 + b**2)/b**2 
     1+ Ke(19,24)*(-a**2 + b**2)/b**2 + Ke(15,18)*(-a**3 + a*b**2)
     1/(b**2*c) + Ke(15,21)*(-a**3 + a*b**2)/(b**2*c) + Ke(18,24)
     1*(-a**3 + a*b**2)/(b**2*c) + Ke(21,24)*(-a**3 + a*b**2)/(b**2*c)
     1+ Ke(15,20)*(a**3 - a*b**2)/b**3 + Ke(15,23)*(a**3 - a*b**2)/b**3 
     1+ Ke(18,23)*(a**3 + 2.0*a*b**2)/b**3 + Ke(19,23)*(2.0*a**2*c 
     1+ 2.0*b**2*c)/b**3 + Ke(20,24)*(a**3 - a*b**2)/b**3 + Ke(21,23)
     1*(a**3 + 2.0*a*b**2)/b**3 + Ke(23,24)*(a**3 - a*b**2)/b**3
      Ke(3,8) =14.0*E*a*c**2/(3.0*b**2*v + 3.0*b**2) - a**4*Ke(18,18)
     1/(b**3*c) - 2.0*a**4*Ke(18,21)/(b**3*c) - a**4*Ke(21,21)/(b**3*c)
     1- 4.0*a**3*Ke(18,19)/b**3 - 4.0*a**3*Ke(18,22)/b**3 
     1- 4.0*a**3*Ke(19,21)/b**3 - 4.0*a**3*Ke(21,22)/b**3 
     1+ a**2*Ke(12,15)/(b*c) + a**2*Ke(12,18)/(b*c) 
     1+ a**2*Ke(12,21)/(b*c) + a**2*Ke(12,24)/(b*c) + 3.0*a**2*Ke(11,18)
     1/b**2 + 3.0*a**2*Ke(11,21)/b**2 - a**2*Ke(18,20)/b**2 
     1 + 3.0*a**2*Ke(18,23)/b**2 - a**2*Ke(20,21)/b**2 
     1+ 3.0*a**2*Ke(21,23)/b**2 - 4.0*a**2*c*Ke(19,19)/b**3 
     1- 8.0*a**2*c*Ke(19,22)/b**3 - 4.0*a**2*c*Ke(22,22)/b**3 
     1+ 2.0*a*Ke(12,19)/b + 2.0*a*Ke(12,22)/b + 6.0*a*c*Ke(11,19)
     1/b**2 + 6.0*a*c*Ke(11,22)/b**2 - 2.0*a*c*Ke(19,20)/b**2 
     1+ 6.0*a*c*Ke(19,23)/b**2 - 2.0*a*c*Ke(20,22)/b**2 
     1+ 6.0*a*c*Ke(22,23)/b**2 - Ke(11,12) + Ke(12,20) - Ke(12,23) 
     1- 2.0*c*Ke(11,11)/b + 2.0*c*Ke(11,20)/b - 4.0*c*Ke(11,23)/b 
     1+ 2.0*c*Ke(20,23)/b - 2.0*c*Ke(23,23)/b + Ke(11,15)*(3.0*a**2
     1- b**2)/b**2 + Ke(11,24)*(3.0*a**2 - b**2)/b**2 + Ke(15,20)
     1*(-a**2 + b**2)/b**2 + Ke(15,23)*(3.0*a**2 - b**2)/b**2 
     1+ Ke(20,24)*(-a**2 + b**2)/b**2 + Ke(23,24)*(3.0*a**2 - b**2)
     1/b**2 + Ke(15,19)*(-4.0*a**3 + 2.0*a*b**2)/b**3 + Ke(15,22)
     1*(-4.0*a**3 + 2.0*a*b**2)/b**3 + Ke(19,24)*(-4.0*a**3 
     1+ 2.0*a*b**2)/b**3 + Ke(22,24)*(-4.0*a**3 + 2.0*a*b**2)/b**3 
     1+ Ke(15,15)*(-a**4 + a**2*b**2)/(b**3*c) + Ke(15,18)*(-2.0*a**4 
     1+ a**2*b**2)/(b**3*c) + Ke(15,21)*(-2.0*a**4 + a**2*b**2)/(b**3*c)
     1+ Ke(15,24)*(-2.0*a**4 + 2.0*a**2*b**2)/(b**3*c) + Ke(18,24)
     1*(-2.0*a**4 + a**2*b**2)/(b**3*c) + Ke(21,24)*(-2.0*a**4 
     1+ a**2*b**2)/(b**3*c) + Ke(24,24)*(-a**4 + a**2*b**2)/(b**3*c)
      Ke(3,09) =8.0*E*a*c/(3.0*b*v + 3.0*b) - a**4*Ke(18,18)/(b**2*c**2)
     1- 4.0*a**3*Ke(18,19)/(b**2*c) - 4.0*a**3*Ke(18,22)/(b**2*c) 
     1+ 2.0*a**2*Ke(11,15)/(b*c) + 2.0*a**2*Ke(11,18)/(b*c) 
     1+ 2.0*a**2*Ke(15,23)/(b*c) + 2.0*a**2*Ke(18,23)/(b*c) 
     1- 4.0*a**2*Ke(19,19)/b**2 - 8.0*a**2*Ke(19,22)/b**2 
     1- 4.0*a**2*Ke(22,22)/b**2 + 4.0*a*Ke(11,19)/b + 4.0*a*Ke(11,22)/b 
     1+ 4.0*a*Ke(19,23)/b + 4.0*a*Ke(22,23)/b + Ke(12,12) 
     1+ 2.0*c*Ke(11,12)/b + 2.0*c*Ke(12,23)/b + Ke(11,21)*(2.0*a**2 
     1+ 2.0*c**2)/(b*c) + Ke(11,24)*(2.0*a**2 + 2.0*c**2)/(b*c) 
     1+ Ke(21,23)*(2.0*a**2 + 2.0*c**2)/(b*c) + Ke(23,24)*(2.0*a**2 
     1+ 2.0*c**2)/(b*c) + Ke(12,19)*(2.0*a*b**2 - 2.0*a*c**2)/(b**2*c) 
     1+ Ke(12,22)*(2.0*a*b**2 - 2.0*a*c**2)/(b**2*c) 
     1+ Ke(15,19)*(-4.0*a**3 + 2.0*a*b**2)/(b**2*c) 
     1+ Ke(15,22)*(-4.0*a**3 + 2.0*a*b**2)/(b**2*c) 
     1+ Ke(19,21)*(-4.0*a**3 - 2.0*a*c**2)/(b**2*c) 
     1+ Ke(19,24)*(-4.0*a**3 + 2.0*a*b**2 - 2.0*a*c**2)/(b**2*c) 
     1+ Ke(21,22)*(-4.0*a**3 - 2.0*a*c**2)/(b**2*c) 
     1+ Ke(22,24)*(-4.0*a**3 + 2.0*a*b**2 - 2.0*a*c**2)/(b**2*c) 
     1+ Ke(12,15)*(a**2*b**2 - a**2*c**2 + b**2*c**2)/(b**2*c**2) 
     1+ Ke(12,18)*(a**2*b**2 - a**2*c**2)/(b**2*c**2) 
     1+ Ke(12,21)*(a**2*b**2 - a**2*c**2 + b**2*c**2)/(b**2*c**2) 
     1+ Ke(12,24)*(a**2*b**2 - a**2*c**2 + 2.0*b**2*c**2)/(b**2*c**2) 
     1+ Ke(15,15)*(-a**4 + a**2*b**2)/(b**2*c**2) 
     1+ Ke(15,18)*(-2.0*a**4 + a**2*b**2)/(b**2*c**2) 
     1+ Ke(15,21)*(-2.0*a**4 + a**2*b**2 - a**2*c**2 
     1+ b**2*c**2)/(b**2*c**2) + Ke(15,24)*(-2.0*a**4 + 2.0*a**2*b**2 
     1- a**2*c**2 + b**2*c**2)/(b**2*c**2) + Ke(18,21)*(-2.0*a**4 
     1- a**2*c**2)/(b**2*c**2) + Ke(18,24)*(-2.0*a**4 + a**2*b**2 
     1- a**2*c**2)/(b**2*c**2) + Ke(21,21)*(-a**4 - a**2*c**2)
     1/(b**2*c**2) + Ke(21,24)*(-2.0*a**4 + a**2*b**2 - 2.0*a**2*c**2 
     1+ b**2*c**2)/(b**2*c**2) + Ke(24,24)*(-a**4 + a**2*b**2 
     1- a**2*c**2 + b**2*c**2)/(b**2*c**2)
      Ke(3,10) = 2.0*E*c**2/(3.0*b*v + 3.0*b) - a**3*Ke(15,18)/(b**2*c) 
     1 - a**3*Ke(15,21)/(b**2*c) - a**3*Ke(18,24)/(b**2*c) 
     1- a**3*Ke(21,24)/(b**2*c) - a**3*Ke(18,20)/b**3 - a**3*Ke(18,23)
     1/b**3 - a**3*Ke(20,21)/b**3 - a**3*Ke(21,23)/b**3 
     1- 2.0*a**2*Ke(15,19)/b**2 - a**2*Ke(18,22)/b**2 
     1- 2.0*a**2*Ke(19,24)/b**2 - a**2*Ke(21,22)/b**2 
     1- 2.0*a**2*c*Ke(19,20)/b**3 - 2.0*a**2*c*Ke(19,23)/b**3 
     1- 2.0*a**2*c*Ke(20,22)/b**3 + a*Ke(12,15)/c + a*Ke(12,24)/c 
     1+ 2.0*a*Ke(11,15)/b + 2.0*a*Ke(11,24)/b + a*Ke(12,20)/b 
     1+ a*Ke(12,23)/b + 2.0*a*c*Ke(11,20)/b**2 + 2.0*a*c*Ke(11,23)/b**2 
     1- 2.0*a*c*Ke(19,22)/b**2 + 2.0*a*c*Ke(20,23)/b**2 
     1- 2.0*a*c*Ke(22,22)/b**2 + 2.0*a*c*Ke(23,23)/b**2 + Ke(12,22) 
     1+ 2.0*c*Ke(11,22)/b + Ke(15,22)*(-3.0*a**2 + b**2)/b**2 
     1+ Ke(22,24)*(-3.0*a**2 + b**2)/b**2 + Ke(15,15)*(-a**3 + a*b**2)
     1/(b**2*c) + Ke(15,24)*(-2.0*a**3 + 2.0*a*b**2)/(b**2*c) 
     1+ Ke(24,24)*(-a**3 + a*b**2)/(b**2*c) + Ke(15,20)*(-a**3 + a*b**2)
     1/b**3 + Ke(15,23)*(-a**3 + 3.0*a*b**2)/b**3 + Ke(20,24)*(-a**3 
     1+ a*b**2)/b**3 + Ke(22,23)*(-2.0*a**2*c + 2.0*b**2*c)/b**3 
     1+ Ke(23,24)*(-a**3 + 3.0*a*b**2)/b**3
      Ke(3,11) = 2.0*E*a*c**2/(3.0*b**2*v + 3.0*b**2) + a**2*Ke(11,18)
     1/b**2 + a**2*Ke(11,21)/b**2 + 2.0*a*c*Ke(11,19)/b**2
     1+ 2.0*a*c*Ke(11,22)/b**2 - Ke(11,12) - 2.0*c*Ke(11,11)/b 
     1- 2.0*c*Ke(11,23)/b + Ke(11,15)*(a**2 - b**2)/b**2 
     1+ Ke(11,24)*(a**2 - b**2)/b**2
      Ke(3,12) = a**2*Ke(12,18)/b**2 + a**2*Ke(12,21)/b**2 
     1+ 2.0*a*c*Ke(12,19)/b**2 + 2.0*a*c*Ke(12,22)/b**2 - Ke(12,12)
     1- 2.0*c*Ke(11,12)/b - 2.0*c*Ke(12,23)/b + Ke(12,15)*(a**2 
     1- b**2)/b**2 + Ke(12,24)*(a**2 - b**2)/b**2
      Ke(3,13) =-E*c**2/(b*v + b) + a**3*Ke(18,18)/(b**2*c) 
     1+ 2.0*a**3*Ke(18,21)/(b**2*c) + a**3*Ke(21,21)/(b**2*c)
     1- a**2*Ke(16,18)/b**2 - a**2*Ke(16,21)/b**2 + 3.0*a**2*Ke(18,19)
     1/b**2 + 3.0*a**2*Ke(18,22)/b**2 + 3.0*a**2*Ke(19,21)/b**2 
     1+ 3.0*a**2*Ke(21,22)/b**2 - a*Ke(12,15)/c - a*Ke(12,18)/c 
     1- a*Ke(12,21)/c - a*Ke(12,24)/c - 2.0*a*Ke(11,15)/b 
     1- 2.0*a*Ke(11,18)/b - 2.0*a*Ke(11,21)/b - 2.0*a*Ke(11,24)/b 
     1- 2.0*a*Ke(15,23)/b - 2.0*a*Ke(18,23)/b - 2.0*a*Ke(21,23)/b 
     1- 2.0*a*Ke(23,24)/b - 2.0*a*c*Ke(16,19)/b**2 - 2.0*a*c*Ke(16,22)
     1/b**2 + 2.0*a*c*Ke(19,19)/b**2 + 4.0*a*c*Ke(19,22)/b**2 
     1+ 2.0*a*c*Ke(22,22)/b**2 + Ke(12,16) - Ke(12,19) - Ke(12,22) 
     1+ 2.0*c*Ke(11,16)/b - 2.0*c*Ke(11,19)/b - 2.0*c*Ke(11,22)/b 
     1+ 2.0*c*Ke(16,23)/b - 2.0*c*Ke(19,23)/b - 2.0*c*Ke(22,23)/b 
     1+ Ke(15,16)*(-a**2 + b**2)/b**2 + Ke(15,19)*(3.0*a**2 - b**2)/b**2
     1+ Ke(15,22)*(3.0*a**2 - b**2)/b**2 + Ke(16,24)*(-a**2 + b**2)/b**2
     1+ Ke(19,24)*(3.0*a**2 - b**2)/b**2 + Ke(22,24)*(3.0*a**2 - b**2)
     1/b**2 + Ke(15,15)*(a**3 - a*b**2)/(b**2*c) + Ke(15,18)*(2.0*a**3 
     1- a*b**2)/(b**2*c) + Ke(15,21)*(2.0*a**3 - a*b**2)/(b**2*c) 
     1+ Ke(15,24)*(2.0*a**3 - 2.0*a*b**2)/(b**2*c) + Ke(18,24)*(2.0*a**3
     1- a*b**2)/(b**2*c) + Ke(21,24)*(2.0*a**3 - a*b**2)/(b**2*c) 
     1+ Ke(24,24)*(a**3 - a*b**2)/(b**2*c)
      Ke(3,14) = E*a*c**2/(3.0*b**2*v + 3.0*b**2) - a**2*Ke(17,18)/b**2 
     1- a**2*Ke(17,21)/b**2 - a**2*Ke(18,20)/b**2 - a**2*Ke(18,23)/b**2 
     1- a**2*Ke(20,21)/b**2 - a**2*Ke(21,23)/b**2-2.0*a*c*Ke(17,19)/b**2
     1- 2.0*a*c*Ke(17,22)/b**2 - 2.0*a*c*Ke(19,20)/b**2 
     1- 2.0*a*c*Ke(19,23)/b**2 - 2.0*a*c*Ke(20,22)/b**2 
     1- 2.0*a*c*Ke(22,23)/b**2 + Ke(12,17) + Ke(12,20) + Ke(12,23) 
     1+ 2.0*c*Ke(11,17)/b + 2.0*c*Ke(11,20)/b + 2.0*c*Ke(11,23)/b 
     1+ 2.0*c*Ke(17,23)/b + 2.0*c*Ke(20,23)/b + 2.0*c*Ke(23,23)/b 
     1+ Ke(15,17)*(-a**2 + b**2)/b**2 + Ke(15,20)*(-a**2 + b**2)/b**2 
     1+ Ke(15,23)*(-a**2 + b**2)/b**2 + Ke(17,24)*(-a**2 + b**2)/b**2 
     1+ Ke(20,24)*(-a**2 + b**2)/b**2 + Ke(23,24)*(-a**2 + b**2)/b**2
      Ke(3,15) = a**2*Ke(15,18)/b**2 + a**2*Ke(15,21)/b**2 
     1+ 2.0*a*c*Ke(15,19)/b**2 + 2.0*a*c*Ke(15,22)/b**2 - Ke(12,15) 
     1- 2.0*c*Ke(11,15)/b - 2.0*c*Ke(15,23)/b + Ke(15,15)*(a**2 
     1- b**2)/b**2 + Ke(15,24)*(a**2 - b**2)/b**2
      Ke(3,16) = -E*c**2/(3.0*b*v + 3.0*b) + a**2*Ke(16,18)/b**2 
     1+ a**2*Ke(16,21)/b**2 + 2.0*a*c*Ke(16,19)/b**2 + 2.0*a*c*Ke(16,22)
     1/b**2 - Ke(12,16) - 2.0*c*Ke(11,16)/b - 2.0*c*Ke(16,23)/b 
     1+ Ke(15,16)*(a**2 - b**2)/b**2 + Ke(16,24)*(a**2 - b**2)/b**2
      Ke(3,17) = -E*a*c**2/(3.0*b**2*v + 3.0*b**2) + a**2*Ke(17,18)/b**2
     1+ a**2*Ke(17,21)/b**2 + 2.0*a*c*Ke(17,19)/b**2 + 2.0*a*c*Ke(17,22)
     1/b**2 - Ke(12,17) - 2.0*c*Ke(11,17)/b - 2.0*c*Ke(17,23)/b 
     1+ Ke(15,17)*(a**2 - b**2)/b**2 + Ke(17,24)*(a**2 - b**2)/b**2
      Ke(3,18) = a**2*Ke(18,18)/b**2 + a**2*Ke(18,21)/b**2 
     1+ 2.0*a*c*Ke(18,19)/b**2 + 2.0*a*c*Ke(18,22)/b**2 - Ke(12,18) 
     1- 2.0*c*Ke(11,18)/b - 2.0*c*Ke(18,23)/b + Ke(15,18)*(a**2 
     1- b**2)/b**2 + Ke(18,24)*(a**2 - b**2)/b**2
      Ke(3,19) = -2.0*E*c**2/(3.0*b*v + 3.0*b) + a**2*Ke(18,19)/b**2 
     1+ a**2*Ke(19,21)/b**2 + 2.0*a*c*Ke(19,19)/b**2 
     1+ 2.0*a*c*Ke(19,22)/b**2 - Ke(12,19) - 2.0*c*Ke(11,19)/b 
     1- 2.0*c*Ke(19,23)/b + Ke(15,19)*(a**2 - b**2)/b**2 
     1+ Ke(19,24)*(a**2 - b**2)/b**2
      Ke(3,20) = -2.0*E*a*c**2/(3.0*b**2*v + 3.0*b**2) + a**2*Ke(18,20)
     1/b**2 + a**2*Ke(20,21)/b**2 + 2.0*a*c*Ke(19,20)/b**2 
     1+ 2.0*a*c*Ke(20,22)/b**2 - Ke(12,20) - 2.0*c*Ke(11,20)/b 
     1- 2.0*c*Ke(20,23)/b + Ke(15,20)*(a**2 - b**2)/b**2 
     1+ Ke(20,24)*(a**2 - b**2)/b**2
      Ke(3,21) = a**2*Ke(18,21)/b**2 + a**2*Ke(21,21)/b**2 
     1+ 2.0*a*c*Ke(19,21)/b**2 + 2.0*a*c*Ke(21,22)/b**2 - Ke(12,21)
     1- 2.0*c*Ke(11,21)/b - 2.0*c*Ke(21,23)/b + Ke(15,21)*(a**2 
     1- b**2)/b**2 + Ke(21,24)*(a**2 - b**2)/b**2
      Ke(3,22) = -2.0*E*c**2/(3.0*b*v + 3.0*b) + a**2*Ke(18,22)/b**2 
     1+ a**2*Ke(21,22)/b**2 + 2.0*a*c*Ke(19,22)/b**2 + 2.0*a*c*Ke(22,22)
     1/b**2 - Ke(12,22) - 2.0*c*Ke(11,22)/b - 2.0*c*Ke(22,23)/b 
     1+ Ke(15,22)*(a**2 - b**2)/b**2 + Ke(22,24)*(a**2 - b**2)/b**2
      Ke(3,23) = 2.0*E*a*c**2/(3.0*b**2*v + 3.0*b**2) + a**2*Ke(18,23)
     1/b**2 + a**2*Ke(21,23)/b**2 + 2.0*a*c*Ke(19,23)/b**2 
     1+ 2.0*a*c*Ke(22,23)/b**2 - Ke(12,23) - 2.0*c*Ke(11,23)/b 
     1- 2.0*c*Ke(23,23)/b + Ke(15,23)*(a**2 - b**2)/b**2 
     1+ Ke(23,24)*(a**2 - b**2)/b**2
      Ke(3,24) = a**2*Ke(18,24)/b**2 + a**2*Ke(21,24)/b**2 
     1+ 2.0*a*c*Ke(19,24)/b**2 + 2.0*a*c*Ke(22,24)/b**2 - Ke(12,24)
     1- 2.0*c*Ke(11,24)/b - 2.0*c*Ke(23,24)/b + Ke(15,24)*(a**2 
     1- b**2)/b**2 + Ke(24,24)*(a**2 - b**2)/b**2
      Ke(4,4) = -E*a*c/(6.0*b*v + 6.0*b) + a**2*Ke(18,18)/c**2 
     1+ 2.0*a**2*Ke(18,21)/c**2 + a**2*Ke(21,21)/c**2 
     1- 2.0*a**2*Ke(18,20)/(b*c) - 2.0*a**2*Ke(18,23)/(b*c) 
     1- 2.0*a**2*Ke(20,21)/(b*c) - 2.0*a**2*Ke(21,23)/(b*c) 
     1+ a**2*Ke(20,20)/b**2 + 2.0*a**2*Ke(20,23)/b**2 + a**2*Ke(23,23)
     1/b**2 - 2.0*a*Ke(16,18)/c - 2.0*a*Ke(16,21)/c + 2.0*a*Ke(16,20)
     1/b + 2.0*a*Ke(16,23)/b + Ke(16,16)
      Ke(4,5)= -E*c/(12.0*v - 12.0) - a*Ke(11,18)/c - a*Ke(11,21)/c 
     1- a*Ke(17,18)/c - a*Ke(17,21)/c - a*Ke(18,23)/c - a*Ke(21,23)/c 
     1+ a*Ke(11,20)/b + a*Ke(11,23)/b + a*Ke(17,20)/b + a*Ke(17,23)/b 
     1+a*Ke(20,23)/b + a*Ke(23,23)/b + Ke(11,16) + Ke(16,17) + Ke(16,23)
      Ke(4,6) =a*Ke(12,18)/c + a*Ke(12,21)/c + 2.0*a*Ke(11,18)/b 
     1+ 2.0*a*Ke(11,21)/b - a*Ke(12,20)/b - a*Ke(12,23)/b 
     1- 2.0*a*c*Ke(11,20)/b**2 - 2.0*a*c*Ke(11,23)/b**2 
     1- 2.0*a*c*Ke(20,23)/b**2 - 2.0*a*c*Ke(23,23)/b**2 - Ke(12,16) 
     1+ (-E*b**2 - E*c**2*v + E*c**2)/(3.0*b*v**2 - 3.0*b) 
     1 - 2.0*c*Ke(11,16)/b - 2.0*c*Ke(16,23)/b + Ke(16,19)*(-2.0*a*b**2 
     1 + 2.0*a*c**2)/(b**2*c) + Ke(16,22)*(-2.0*a*b**2 + 2.0*a*c**2)
     1/(b**2*c) + Ke(15,16)*(-a**2*b**2 + a**2*c**2)/(b**2*c**2) 
     1+ Ke(16,18)*(-a**2*b**2 + a**2*c**2 + b**2*c**2)/(b**2*c**2) 
     1+ Ke(16,21)*(-a**2*b**2 + a**2*c**2)/(b**2*c**2) + Ke(16,24)
     1*(-a**2*b**2 + a**2*c**2 - b**2*c**2)/(b**2*c**2) + Ke(18,19)
     1*(2.0*a**2*b**2 - 2.0*a**2*c**2)/(b**2*c**2) + Ke(18,22)
     1*(2.0*a**2*b**2 - 2.0*a**2*c**2)/(b**2*c**2) + Ke(19,21)
     1*(2.0*a**2*b**2 - 2.0*a**2*c**2)/(b**2*c**2) + Ke(21,22)
     1*(2.0*a**2*b**2 - 2.0*a**2*c**2)/(b**2*c**2) + Ke(15,18)
     1*(a**3*b**2 - a**3*c**2)/(b**2*c**3) + Ke(15,21)*(a**3*b**2 
     1- a**3*c**2)/(b**2*c**3) + Ke(18,18)*(a**3*b**2 - a**3*c**2
     1- a*b**2*c**2)/(b**2*c**3) + Ke(18,21)*(2.0*a**3*b**2 
     1- 2.0*a**3*c**2 - a*b**2*c**2)/(b**2*c**3) + Ke(18,24)
     1*(a**3*b**2 - a**3*c**2 + a*b**2*c**2)/(b**2*c**3) + Ke(21,21)
     1*(a**3*b**2 - a**3*c**2)/(b**2*c**3) + Ke(21,24)*(a**3*b**2 
     1- a**3*c**2 + a*b**2*c**2)/(b**2*c**3) + Ke(19,20)*(-2.0*a**2*b**2
     1+ 2.0*a**2*c**2)/(b**3*c) + Ke(19,23)*(-2.0*a**2*b**2 
     1+ 2.0*a**2*c**2)/(b**3*c) + Ke(20,22)*(-2.0*a**2*b**2 
     1+ 2.0*a**2*c**2)/(b**3*c) + Ke(22,23)*(-2.0*a**2*b**2 
     1+ 2.0*a**2*c**2)/(b**3*c) + Ke(15,20)*(-a**3*b**2 + a**3*c**2)
     1/(b**3*c**2) + Ke(15,23)*(-a**3*b**2 + a**3*c**2)/(b**3*c**2) 
     1+ Ke(18,20)*(-a**3*b**2 + a**3*c**2 + a*b**2*c**2)/(b**3*c**2)
     1+ Ke(18,23)*(-a**3*b**2 + a**3*c**2 + 3.0*a*b**2*c**2)/(b**3*c**2) 
     1+ Ke(20,21)*(-a**3*b**2 + a**3*c**2)/(b**3*c**2) + Ke(20,24)
     1*(-a**3*b**2 + a**3*c**2 - a*b**2*c**2)/(b**3*c**2) + Ke(21,23)
     1*(-a**3*b**2 + a**3*c**2 + 2.0*a*b**2*c**2)/(b**3*c**2) 
     1+ Ke(23,24)*(-a**3*b**2 + a**3*c**2 - a*b**2*c**2)/(b**3*c**2)
      Ke(4,7) = 5.0*E*a*c/(12.0*b*v + 12.0*b) - a**2*Ke(18,18)/c**2 
     1- 2.0*a**2*Ke(18,21)/c**2 - a**2*Ke(21,21)/c**2 
     1+ 2.0*a**2*Ke(18,20)/(b*c) + 2.0*a**2*Ke(18,23)/(b*c) 
     1+ 2.0*a**2*Ke(20,21)/(b*c) + 2.0*a**2*Ke(21,23)/(b*c) 
     1- a**2*Ke(20,20)/b**2 - 2.0*a**2*Ke(20,23)/b**2 
     1- a**2*Ke(23,23)/b**2 + a*Ke(16,18)/c + a*Ke(16,21)/c 
     1- a*Ke(18,19)/c - a*Ke(19,21)/c - a*Ke(16,20)/b - a*Ke(16,23)/b 
     1+ a*Ke(19,20)/b + a*Ke(19,23)/b + Ke(16,19)
      Ke(4,8) = -5.0*E*c/(12.0*v + 12.0) - a**3*Ke(15,18)/(b*c**2) 
     1- a**3*Ke(15,21)/(b*c**2) - a**3*Ke(18,18)/(b*c**2) 
     1 - 2.0*a**3*Ke(18,21)/(b*c**2) - a**3*Ke(18,24)/(b*c**2) 
     1- a**3*Ke(21,21)/(b*c**2) - a**3*Ke(21,24)/(b*c**2) 
     1+ a**3*Ke(15,20)/(b**2*c) + a**3*Ke(15,23)/(b**2*c) 
     1 + a**3*Ke(20,24)/(b**2*c) + a**3*Ke(23,24)/(b**2*c) 
     1+ a**2*Ke(15,16)/(b*c) + a**2*Ke(16,18)/(b*c) + a**2*Ke(16,21)
     1/(b*c) + a**2*Ke(16,24)/(b*c) - 2.0*a**2*Ke(18,19)/(b*c) 
     1 - 2.0*a**2*Ke(18,22)/(b*c) - 2.0*a**2*Ke(19,21)/(b*c) 
     1- 2.0*a**2*Ke(21,22)/(b*c) + 2.0*a**2*Ke(19,20)/b**2 
     1+ 2.0*a**2*Ke(19,23)/b**2 + 2.0*a**2*Ke(20,22)/b**2 
     1+ 2.0*a**2*Ke(22,23)/b**2 + a*Ke(11,18)/c + a*Ke(11,21)/c 
     1- a*Ke(11,20)/b - a*Ke(11,23)/b + 2.0*a*Ke(16,19)/b 
     1+ 2.0*a*Ke(16,22)/b + a*Ke(20,20)/b - a*Ke(23,23)/b - Ke(11,16) 
     1+ Ke(16,20) - Ke(16,23) + Ke(18,20)*(a**3 - a*b**2)/(b**2*c) 
     1+ Ke(18,23)*(a**3 + a*b**2)/(b**2*c) + Ke(20,21)*(a**3 - a*b**2)
     1/(b**2*c) + Ke(21,23)*(a**3 + a*b**2)/(b**2*c)
      Ke(4,9) = E*b/(3.0*v**2 - 3.0) - a**3*Ke(15,18)/c**3 
     1- a**3*Ke(15,21)/c**3 - a**3*Ke(18,18)/c**3 + a**3*Ke(15,20)
     1/(b*c**2) + a**3*Ke(15,23)/(b*c**2) + a**3*Ke(18,20)/(b*c**2) 
     1+ a**3*Ke(18,23)/(b*c**2) + a**2*Ke(15,16)/c**2 + a**2*Ke(16,18)
     1/c**2 - 2.0*a**2*Ke(18,19)/c**2 - 2.0*a**2*Ke(18,22)/c**2 
     1- 2.0*a**2*Ke(19,21)/c**2 - 2.0*a**2*Ke(21,22)/c**2 
     1+ 2.0*a**2*Ke(19,20)/(b*c) + 2.0*a**2*Ke(19,23)/(b*c) 
     1+ 2.0*a**2*Ke(20,22)/(b*c) + 2.0*a**2*Ke(22,23)/(b*c) 
     1- a*Ke(12,18)/c - a*Ke(12,21)/c + 2.0*a*Ke(16,19)/c 
     1+ 2.0*a*Ke(16,22)/c + a*Ke(12,20)/b + a*Ke(12,23)/b 
     1+ Ke(12,16) + Ke(16,21)*(a**2 + c**2)/c**2 + Ke(16,24)
     1*(a**2 + c**2)/c**2 + Ke(18,21)*(-2.0*a**3 - a*c**2)/c**3 
     1+ Ke(18,24)*(-a**3 - a*c**2)/c**3 + Ke(21,21)*(-a**3 - a*c**2)
     1/c**3 + Ke(21,24)*(-a**3 - a*c**2)/c**3 + Ke(20,21)
     1*(a**3 + a*c**2)/(b*c**2) + Ke(20,24)*(a**3 + a*c**2)/(b*c**2)
     1+ Ke(21,23)*(a**3 + a*c**2)/(b*c**2) 
     1+ Ke(23,24)*(a**3 + a*c**2)/(b*c**2)
      Ke(4,10) = -5.0*E*a*c/(12.0*b*v + 12.0*b) - a**2*Ke(15,18)/c**2 
     1- a**2*Ke(15,21)/c**2 - a**2*Ke(18,24)/c**2 - a**2*Ke(21,24)/c**2 
     1+ a**2*Ke(15,20)/(b*c) + a**2*Ke(15,23)/(b*c)-a**2*Ke(18,20)/(b*c)
     1- a**2*Ke(18,23)/(b*c) - a**2*Ke(20,21)/(b*c)+a**2*Ke(20,24)/(b*c)
     1- a**2*Ke(21,23)/(b*c) + a**2*Ke(23,24)/(b*c)+a**2*Ke(20,20)/b**2 
     1+ 2.0*a**2*Ke(20,23)/b**2 + a**2*Ke(23,23)/b**2 + a*Ke(15,16)/c 
     1+ a*Ke(16,24)/c - a*Ke(18,22)/c - a*Ke(21,22)/c + a*Ke(16,20)/b 
     1+ a*Ke(16,23)/b + a*Ke(20,22)/b + a*Ke(22,23)/b + Ke(16,22)
      Ke(4,11) = -E*c/(12.0*v + 12.0) + a*Ke(11,18)/c + a*Ke(11,21)/c 
     1- a*Ke(11,20)/b - a*Ke(11,23)/b - Ke(11,16)
      Ke(4,12) = a*Ke(12,18)/c + a*Ke(12,21)/c - a*Ke(12,20)/b 
     1- a*Ke(12,23)/b - Ke(12,16)
      Ke(4,13) = a**2*Ke(15,18)/c**2 + a**2*Ke(15,21)/c**2 
     1+ a**2*Ke(18,18)/c**2 + 2.0*a**2*Ke(18,21)/c**2 
     1+ a**2*Ke(18,24)/c**2 + a**2*Ke(21,21)/c**2 + a**2*Ke(21,24)/c**2 
     1- a**2*Ke(15,20)/(b*c) - a**2*Ke(15,23)/(b*c) 
     1- a**2*Ke(18,20)/(b*c) - a**2*Ke(18,23)/(b*c)-a**2*Ke(20,21)/(b*c)
     1- a**2*Ke(20,24)/(b*c) - a**2*Ke(21,23)/(b*c)-a**2*Ke(23,24)/(b*c)
     1- a*Ke(15,16)/c - 2.0*a*Ke(16,18)/c - 2.0*a*Ke(16,21)/c 
     1- a*Ke(16,24)/c + a*Ke(18,19)/c + a*Ke(18,22)/c + a*Ke(19,21)/c 
     1+ a*Ke(21,22)/c + a*Ke(16,20)/b + a*Ke(16,23)/b - a*Ke(19,20)/b
     1- a*Ke(19,23)/b - a*Ke(20,22)/b - a*Ke(22,23)/b + Ke(16,16) 
     1- Ke(16,19) - Ke(16,22) + (-E*a**2*c*v + E*a**2*c + 2.0*E*b**2*c)
     1/(12.0*a*b*v**2 - 12.0*a*b)
      Ke(4,14) = -E*c/(3.0*v + 3.0) - a*Ke(17,18)/c - a*Ke(17,21)/c 
     1- a*Ke(18,20)/c - a*Ke(18,23)/c - a*Ke(20,21)/c - a*Ke(21,23)/c 
     1+ a*Ke(17,20)/b + a*Ke(17,23)/b + a*Ke(20,20)/b + 2.0*a*Ke(20,23)
     1/b + a*Ke(23,23)/b + Ke(16,17) + Ke(16,20) + Ke(16,23)
      Ke(4,15) = a*Ke(15,18)/c + a*Ke(15,21)/c - a*Ke(15,20)/b 
     1- a*Ke(15,23)/b - Ke(15,16)
      Ke(4,16) = E*a*c/(12.0*b*v + 12.0*b) + a*Ke(16,18)/c 
     1+ a*Ke(16,21)/c - a*Ke(16,20)/b - a*Ke(16,23)/b - Ke(16,16)
      Ke(4,17) = -E*c/(12.0*v + 12.0) + a*Ke(17,18)/c + a*Ke(17,21)/c 
     1- a*Ke(17,20)/b - a*Ke(17,23)/b - Ke(16,17)
      Ke(4,18) = a*Ke(18,18)/c + a*Ke(18,21)/c - a*Ke(18,20)/b 
     1- a*Ke(18,23)/b - Ke(16,18)
      Ke(4,19) = -E*a*c/(12.0*b*v + 12.0*b) + a*Ke(18,19)/c 
     1+ a*Ke(19,21)/c - a*Ke(19,20)/b - a*Ke(19,23)/b - Ke(16,19)
      Ke(4,20) = E*c/(12.0*v + 12.0) + a*Ke(18,20)/c + a*Ke(20,21)/c 
     1- a*Ke(20,20)/b - a*Ke(20,23)/b - Ke(16,20)
      Ke(4,21) = a*Ke(18,21)/c + a*Ke(21,21)/c - a*Ke(20,21)/b 
     1- a*Ke(21,23)/b - Ke(16,21)
      Ke(4,22) = E*a*c/(12.0*b*v + 12.0*b) + a*Ke(18,22)/c 
     1+ a*Ke(21,22)/c - a*Ke(20,22)/b - a*Ke(22,23)/b - Ke(16,22)
      Ke(4,23) = E*c/(12.0*v + 12.0) + a*Ke(18,23)/c + a*Ke(21,23)/c 
     1- a*Ke(20,23)/b - a*Ke(23,23)/b - Ke(16,23)
      Ke(4,24) = a*Ke(18,24)/c + a*Ke(21,24)/c - a*Ke(20,24)/b 
     1- a*Ke(23,24)/b - Ke(16,24)
      Ke(5,5) = E*a*c/(3.0*b*v**2 - 3.0*b) + Ke(11,11) + 2.0*Ke(11,17) 
     1+ 2.0*Ke(11,23) + Ke(17,17) + 2.0*Ke(17,23) + Ke(23,23)
      Ke(5,6) = -Ke(11,12) - Ke(12,17) - Ke(12,23) + (-E*a*b**2*v 
     1+ E*a*c**2*v - E*a*c**2)/(b**2*v**2 - b**2) - 2.0*c*Ke(11,11)/b 
     1- 2.0*c*Ke(11,17)/b - 4.0*c*Ke(11,23)/b - 2.0*c*Ke(17,23)/b 
     1- 2.0*c*Ke(23,23)/b + Ke(11,19)*(-2.0*a*b**2 +2.0*a*c**2)/(b**2*c)
     1+ Ke(11,22)*(-2.0*a*b**2 + 2.0*a*c**2)/(b**2*c) + Ke(17,19)
     1*(-2.0*a*b**2 + 2.0*a*c**2)/(b**2*c) + Ke(17,22)*(-2.0*a*b**2
     1+ 2.0*a*c**2)/(b**2*c) + Ke(19,23)*(-2.0*a*b**2 + 2.0*a*c**2)
     1/(b**2*c) + Ke(22,23)*(-2.0*a*b**2 + 2.0*a*c**2)/(b**2*c)
     1+ Ke(11,15)*(-a**2*b**2 + a**2*c**2)/(b**2*c**2) + Ke(11,18)
     1*(-a**2*b**2 + a**2*c**2 + b**2*c**2)/(b**2*c**2) + Ke(11,21)
     1*(-a**2*b**2 + a**2*c**2)/(b**2*c**2) + Ke(11,24)*(-a**2*b**2 
     1+ a**2*c**2 - b**2*c**2)/(b**2*c**2) + Ke(15,17)*(-a**2*b**2 
     1+ a**2*c**2)/(b**2*c**2) + Ke(15,23)*(-a**2*b**2 + a**2*c**2)
     1/(b**2*c**2) + Ke(17,18)*(-a**2*b**2 + a**2*c**2 + b**2*c**2)
     1/(b**2*c**2) + Ke(17,21)*(-a**2*b**2 + a**2*c**2)/(b**2*c**2)
     1+ Ke(17,24)*(-a**2*b**2 + a**2*c**2 - b**2*c**2)/(b**2*c**2) 
     1+ Ke(18,23)*(-a**2*b**2 + a**2*c**2 + b**2*c**2)/(b**2*c**2) 
     1+ Ke(21,23)*(-a**2*b**2 + a**2*c**2)/(b**2*c**2) + Ke(23,24)
     1*(-a**2*b**2 + a**2*c**2 - b**2*c**2)/(b**2*c**2)
      Ke(5,7) = a*Ke(11,18)/c + a*Ke(11,21)/c + a*Ke(17,18)/c 
     1+ a*Ke(17,21)/c + a*Ke(18,23)/c + a*Ke(21,23)/c - a*Ke(11,20)/b 
     1- a*Ke(11,23)/b - a*Ke(17,20)/b - a*Ke(17,23)/b - a*Ke(20,23)/b 
     1- a*Ke(23,23)/b + Ke(11,19) + Ke(17,19) + Ke(19,23) + (2.0*E*c*v
     1- E*c)/(6.0*v**2 - 6.0)
      Ke(5,8) = a**2*Ke(11,15)/(b*c) + a**2*Ke(11,18)/(b*c) 
     1+ a**2*Ke(11,21)/(b*c) + a**2*Ke(11,24)/(b*c) 
     1+ a**2*Ke(15,17)/(b*c) + a**2*Ke(15,23)/(b*c) + a**2*Ke(17,18)
     1/(b*c) + a**2*Ke(17,21)/(b*c) + a**2*Ke(17,24)/(b*c) 
     1 + a**2*Ke(18,23)/(b*c) + a**2*Ke(21,23)/(b*c) + a**2*Ke(23,24)
     1/(b*c) + 2.0*a*Ke(11,19)/b + 2.0*a*Ke(11,22)/b + 2.0*a*Ke(17,19)/b
     1+ 2.0*a*Ke(17,22)/b + 2.0*a*Ke(19,23)/b + 2.0*a*Ke(22,23)/b 
     1- Ke(11,11) - Ke(11,17) + Ke(11,20) - 2.0*Ke(11,23) + Ke(17,20) 
     1- Ke(17,23) + Ke(20,23) - Ke(23,23) + (6.0*E*a*c*v - 5.0*E*a*c)
     1/(6.0*b*v**2 - 6.0*b)
      Ke(5,9)= E*a*v/(v**2 - 1.0) + a**2*Ke(11,15)/c**2 + a**2*Ke(11,18)
     1/c**2 + a**2*Ke(15,17)/c**2 + a**2*Ke(15,23)/c**2 + a**2*Ke(17,18)
     1/c**2 + a**2*Ke(18,23)/c**2 + 2.0*a*Ke(11,19)/c + 2.0*a*Ke(11,22)
     1/c + 2.0*a*Ke(17,19)/c + 2.0*a*Ke(17,22)/c + 2.0*a*Ke(19,23)/c 
     1+ 2.0*a*Ke(22,23)/c + Ke(11,12) + Ke(12,17) + Ke(12,23) 
     1+ Ke(11,21)*(a**2 + c**2)/c**2 + Ke(11,24)*(a**2 + c**2)/c**2 
     1+ Ke(17,21)*(a**2 + c**2)/c**2 + Ke(17,24)*(a**2 + c**2)/c**2 
     1+ Ke(21,23)*(a**2 + c**2)/c**2 + Ke(23,24)*(a**2 + c**2)/c**2
      Ke(5,10) = E*c/(6.0*v**2 - 6.0) + a*Ke(11,15)/c + a*Ke(11,24)/c 
     1+ a*Ke(15,17)/c + a*Ke(15,23)/c + a*Ke(17,24)/c + a*Ke(23,24)/c 
     1+ a*Ke(11,20)/b + a*Ke(11,23)/b + a*Ke(17,20)/b + a*Ke(17,23)/b 
     1+ a*Ke(20,23)/b + a*Ke(23,23)/b + Ke(11,22) + Ke(17,22) +Ke(22,23)
      Ke(5,11) = -E*a*c/(6.0*b*v**2 - 6.0*b) - Ke(11,11) 
     1- Ke(11,17) - Ke(11,23)
      Ke(5,12) = -Ke(11,12) - Ke(12,17) - Ke(12,23)
      Ke(5,13) = -E*c*v/(3.0*v**2 - 3.0) - a*Ke(11,15)/c - a*Ke(11,18)/c
     1- a*Ke(11,21)/c - a*Ke(11,24)/c - a*Ke(15,17)/c - a*Ke(15,23)/c 
     1- a*Ke(17,18)/c - a*Ke(17,21)/c - a*Ke(17,24)/c - a*Ke(18,23)/c 
     1- a*Ke(21,23)/c - a*Ke(23,24)/c + Ke(11,16) - Ke(11,19) 
     1- Ke(11,22) + Ke(16,17) + Ke(16,23) - Ke(17,19) - Ke(17,22) 
     1- Ke(19,23) - Ke(22,23)
      Ke(5,14) = Ke(11,17) + Ke(11,20) + Ke(11,23) + Ke(17,17) 
     1+ Ke(17,20) + 2.0*Ke(17,23) + Ke(20,23) + Ke(23,23) 
     1+ (2.0*E*a**2*c - 3.0*E*b**2*c*v + 3.0*E*b**2*c)
     1/(12.0*a*b*v**2 - 12.0*a*b)
      Ke(5,15) = -Ke(11,15) - Ke(15,17) - Ke(15,23)
      Ke(5,16) = E*c*v/(6.0*v**2 - 6.0) - Ke(11,16) 
     1- Ke(16,17) - Ke(16,23)
      Ke(5,17) = -E*a*c/(6.0*b*v**2 - 6.0*b) - Ke(11,17) 
     1- Ke(17,17) - Ke(17,23)
      Ke(5,18) = -Ke(11,18) - Ke(17,18) - Ke(18,23)
      Ke(5,19) = -E*c*v/(6.0*v**2 - 6.0) - Ke(11,19) 
     1- Ke(17,19) - Ke(19,23)
      Ke(5,20) = E*a*c/(6.0*b*v**2 - 6.0*b) - Ke(11,20) 
     1- Ke(17,20) - Ke(20,23)
      Ke(5,21) = -Ke(11,21) - Ke(17,21) - Ke(21,23)
      Ke(5,22) = -E*c*v/(6.0*v**2 - 6.0) - Ke(11,22) 
     1- Ke(17,22) - Ke(22,23)
      Ke(5,23) = -E*a*c/(6.0*b*v**2 - 6.0*b) - Ke(11,23)
     1- Ke(17,23) - Ke(23,23)
      Ke(5,24) = -Ke(11,24) - Ke(17,24) - Ke(23,24)
      Ke(6,6) = Ke(12,12) + (8.0*E*a*b**4 + 16.0*E*a*b**2*c**2*v 
     1- 16.0*E*a*b**2*c**2 - 16.0*E*a*c**4*v + 16.0*E*a*c**4)
     1/(3.0*b**3*c*v**2 - 3.0*b**3*c) + 4.0*c*Ke(11,12)/b 
     1+ 4.0*c*Ke(12,23)/b + 4.0*c**2*Ke(11,11)/b**2 
     1+ 8.0*c**2*Ke(11,23)/b**2 + 4.0*c**2*Ke(23,23)/b**2 + Ke(12,19)
     1*(4.0*a*b**2 - 4.0*a*c**2)/(b**2*c) + Ke(12,22)*(4.0*a*b**2 
     1- 4.0*a*c**2)/(b**2*c) + Ke(12,15)*(2.0*a**2*b**2 - 2.0*a**2*c**2)
     1/(b**2*c**2) + Ke(12,18)*(2.0*a**2*b**2 - 2.0*a**2*c**2 
     1- 2.0*b**2*c**2)/(b**2*c**2) + Ke(12,21)*(2.0*a**2*b**2 
     1- 2.0*a**2*c**2)/(b**2*c**2) + Ke(12,24)*(2.0*a**2*b**2
     1 - 2.0*a**2*c**2 + 2.0*b**2*c**2)/(b**2*c**2) + Ke(11,19)
     1*(8.0*a*b**2 - 8.0*a*c**2)/b**3 + Ke(11,22)*(8.0*a*b**2 
     1- 8.0*a*c**2)/b**3 + Ke(19,23)*(8.0*a*b**2 - 8.0*a*c**2)
     1/b**3 + Ke(22,23)*(8.0*a*b**2 - 8.0*a*c**2)/b**3 + Ke(11,15)
     1*(4.0*a**2*b**2 - 4.0*a**2*c**2)/(b**3*c) + Ke(11,18)
     1*(4.0*a**2*b**2 - 4.0*a**2*c**2 - 4.0*b**2*c**2)/(b**3*c) 
     1+ Ke(11,21)*(4.0*a**2*b**2 - 4.0*a**2*c**2)/(b**3*c) 
     1+ Ke(11,24)*(4.0*a**2*b**2 - 4.0*a**2*c**2 + 4.0*b**2*c**2)
     1/(b**3*c) + Ke(15,23)*(4.0*a**2*b**2 - 4.0*a**2*c**2)/(b**3*c) 
     1+ Ke(18,23)*(4.0*a**2*b**2 - 4.0*a**2*c**2 - 4.0*b**2*c**2)
     1/(b**3*c) + Ke(21,23)*(4.0*a**2*b**2 - 4.0*a**2*c**2)/(b**3*c) 
     1+ Ke(23,24)*(4.0*a**2*b**2 - 4.0*a**2*c**2 + 4.0*b**2*c**2)
     1/(b**3*c) + Ke(19,19)*(4.0*a**2*b**4 - 8.0*a**2*b**2*c**2 
     1+ 4.0*a**2*c**4)/(b**4*c**2) + Ke(19,22)*(8.0*a**2*b**4 
     1- 16.0*a**2*b**2*c**2 + 8.0*a**2*c**4)/(b**4*c**2)
     1+ Ke(22,22)*(4.0*a**2*b**4 - 8.0*a**2*b**2*c**2 + 4.0*a**2*c**4)
     1/(b**4*c**2) + Ke(15,19)*(4.0*a**3*b**4 - 8.0*a**3*b**2*c**2 
     1+ 4.0*a**3*c**4)/(b**4*c**3) + Ke(15,22)*(4.0*a**3*b**4 
     1 - 8.0*a**3*b**2*c**2 + 4.0*a**3*c**4)/(b**4*c**3) + Ke(18,19)
     1*(4.0*a**3*b**4 - 8.0*a**3*b**2*c**2 + 4.0*a**3*c**4 
     1- 4.0*a*b**4*c**2 + 4.0*a*b**2*c**4)/(b**4*c**3) + Ke(18,22)
     1*(4.0*a**3*b**4 - 8.0*a**3*b**2*c**2 + 4.0*a**3*c**4 
     1- 4.0*a*b**4*c**2 + 4.0*a*b**2*c**4)/(b**4*c**3) + Ke(19,21)
     1*(4.0*a**3*b**4 - 8.0*a**3*b**2*c**2 + 4.0*a**3*c**4)
     1/(b**4*c**3) + Ke(19,24)*(4.0*a**3*b**4 - 8.0*a**3*b**2*c**2
     1+ 4.0*a**3*c**4 + 4.0*a*b**4*c**2 - 4.0*a*b**2*c**4)/(b**4*c**3)
     1+ Ke(21,22)*(4.0*a**3*b**4 - 8.0*a**3*b**2*c**2 + 4.0*a**3*c**4)
     1/(b**4*c**3) + Ke(22,24)*(4.0*a**3*b**4 - 8.0*a**3*b**2*c**2
     1+ 4.0*a**3*c**4 + 4.0*a*b**4*c**2 - 4.0*a*b**2*c**4)/(b**4*c**3)
     1+ Ke(15,15)*(a**4*b**4-2.0*a**4*b**2*c**2+a**4*c**4)/(b**4*c**4) 
     1+ Ke(15,18)*(2.0*a**4*b**4 - 4.0*a**4*b**2*c**2 + 2.0*a**4*c**4 
     1- 2.0*a**2*b**4*c**2 + 2.0*a**2*b**2*c**4)/(b**4*c**4) 
     1+ Ke(15,21)*(2.0*a**4*b**4 - 4.0*a**4*b**2*c**2 + 2.0*a**4*c**4)
     1/(b**4*c**4) + Ke(15,24)*(2.0*a**4*b**4 - 4.0*a**4*b**2*c**2 
     1+ 2.0*a**4*c**4 + 2.0*a**2*b**4*c**2 - 2.0*a**2*b**2*c**4)
     1/(b**4*c**4) + Ke(18,18)*(a**4*b**4 - 2.0*a**4*b**2*c**2 
     1+ a**4*c**4 - 2.0*a**2*b**4*c**2 + 2.0*a**2*b**2*c**4
     1+ b**4*c**4)/(b**4*c**4) + Ke(18,21)*(2.0*a**4*b**4 
     1- 4.0*a**4*b**2*c**2 + 2.0*a**4*c**4 - 2.0*a**2*b**4*c**2 
     1+ 2.0*a**2*b**2*c**4)/(b**4*c**4) + Ke(18,24)*(2.0*a**4*b**4 
     1- 4.0*a**4*b**2*c**2 + 2.0*a**4*c**4 - 2.0*b**4*c**4)/(b**4*c**4)
     1+ Ke(21,21)*(a**4*b**4 - 2.0*a**4*b**2*c**2 + a**4*c**4)
     1/(b**4*c**4) + Ke(21,24)*(2.0*a**4*b**4 - 4.0*a**4*b**2*c**2 
     1+ 2.0*a**4*c**4 + 2.0*a**2*b**4*c**2 - 2.0*a**2*b**2*c**4)
     1/(b**4*c**4) + Ke(24,24)*(a**4*b**4 - 2.0*a**4*b**2*c**2 
     1+ a**4*c**4 + 2.0*a**2*b**4*c**2 - 2.0*a**2*b**2*c**4 
     1+ b**4*c**4)/(b**4*c**4)
      Ke(6,7) = -a*Ke(12,18)/c - a*Ke(12,21)/c - 2.0*a*Ke(11,18)/b 
     1- 2.0*a*Ke(11,21)/b + a*Ke(12,20)/b + a*Ke(12,23)/b 
     1+ 2.0*a*c*Ke(11,20)/b**2 + 2.0*a*c*Ke(11,23)/b**2 
     1+ 2.0*a*c*Ke(20,23)/b**2 + 2.0*a*c*Ke(23,23)/b**2 - Ke(12,19) 
     1+ (-2.0*E*b**2 - 2.0*E*c**2*v + 2.0*E*c**2)/(3.0*b*v**2 - 3.0*b) 
     1- 2.0*c*Ke(11,19)/b + Ke(19,19)*(-2.0*a*b**2 +2.0*a*c**2)/(b**2*c)
     1+ Ke(19,22)*(-2.0*a*b**2 + 2.0*a*c**2)/(b**2*c) 
     1+ Ke(15,19)*(-a**2*b**2 + a**2*c**2)/(b**2*c**2) 
     1+ Ke(18,19)*(-3.0*a**2*b**2 + 3.0*a**2*c**2 + b**2*c**2)
     1/(b**2*c**2) + Ke(18,22)*(-2.0*a**2*b**2 + 2.0*a**2*c**2)
     1/(b**2*c**2) + Ke(19,21)*(-3.0*a**2*b**2 + 3.0*a**2*c**2)
     1/(b**2*c**2) + Ke(19,24)*(-a**2*b**2 + a**2*c**2 - b**2*c**2)
     1/(b**2*c**2) + Ke(21,22)*(-2.0*a**2*b**2 + 2.0*a**2*c**2)
     1/(b**2*c**2) + Ke(15,18)*(-a**3*b**2 + a**3*c**2)/(b**2*c**3) 
     1+ Ke(15,21)*(-a**3*b**2 + a**3*c**2)/(b**2*c**3) + Ke(18,18)
     1*(-a**3*b**2 + a**3*c**2 + a*b**2*c**2)/(b**2*c**3) + Ke(18,21)
     1*(-2.0*a**3*b**2 + 2.0*a**3*c**2 + a*b**2*c**2)/(b**2*c**3) 
     1+ Ke(18,24)*(-a**3*b**2 + a**3*c**2 - a*b**2*c**2)/(b**2*c**3)
     1+ Ke(21,21)*(-a**3*b**2 + a**3*c**2)/(b**2*c**3) + Ke(21,24)
     1*(-a**3*b**2 + a**3*c**2 - a*b**2*c**2)/(b**2*c**3) + Ke(19,20)
     1*(2.0*a**2*b**2 - 2.0*a**2*c**2)/(b**3*c) 
     1+ Ke(19,23)*(2.0*a**2*b**2 - 2.0*a**2*c**2 - 2.0*b**2*c**2)
     1/(b**3*c) + Ke(20,22)*(2.0*a**2*b**2 - 2.0*a**2*c**2)/(b**3*c) 
     1+ Ke(22,23)*(2.0*a**2*b**2 - 2.0*a**2*c**2)/(b**3*c) + Ke(15,20)
     1*(a**3*b**2 - a**3*c**2)/(b**3*c**2) + Ke(15,23)*(a**3*b**2 
     1- a**3*c**2)/(b**3*c**2) + Ke(18,20)*(a**3*b**2 - a**3*c**2 
     1- a*b**2*c**2)/(b**3*c**2) + Ke(18,23)*(a**3*b**2 - a**3*c**2 
     1- 3.0*a*b**2*c**2)/(b**3*c**2) + Ke(20,21)*(a**3*b**2 
     1- a**3*c**2)/(b**3*c**2) + Ke(20,24)*(a**3*b**2 - a**3*c**2 
     1+ a*b**2*c**2)/(b**3*c**2) + Ke(21,23)*(a**3*b**2 - a**3*c**2 
     1- 2.0*a*b**2*c**2)/(b**3*c**2) + Ke(23,24)*(a**3*b**2 
     1- a**3*c**2 + a*b**2*c**2)/(b**3*c**2)
      Ke(6,8) = -a**2*Ke(12,15)/(b*c) - a**2*Ke(12,18)/(b*c) 
     1- a**2*Ke(12,21)/(b*c) - a**2*Ke(12,24)/(b*c) - 2.0*a*Ke(12,19)/b 
     1- 2.0*a*Ke(12,22)/b + Ke(11,12) - Ke(12,20) + Ke(12,23) 
     1+ (6.0*E*a*b**2*v - 8.0*E*a*b**2 - 14.0*E*a*c**2*v 
     1+ 14.0*E*a*c**2)/(3.0*b**2*v**2 - 3.0*b**2) + 2.0*c*Ke(11,11)/b 
     1- 2.0*c*Ke(11,20)/b + 4.0*c*Ke(11,23)/b - 2.0*c*Ke(20,23)/b 
     1+ 2.0*c*Ke(23,23)/b + Ke(11,19)*(2.0*a*b**2 - 6.0*a*c**2)/(b**2*c)
     1+ Ke(11,22)*(2.0*a*b**2 - 6.0*a*c**2)/(b**2*c) + Ke(19,20)
     1*(-2.0*a*b**2 + 2.0*a*c**2)/(b**2*c) + Ke(19,23)*(2.0*a*b**2
     1- 6.0*a*c**2)/(b**2*c) + Ke(20,22)*(-2.0*a*b**2 + 2.0*a*c**2)
     1/(b**2*c) + Ke(22,23)*(2.0*a*b**2 - 6.0*a*c**2)/(b**2*c) 
     1+ Ke(11,15)*(a**2*b**2 - 3.0*a**2*c**2)/(b**2*c**2) + Ke(11,18)
     1*(a**2*b**2 - 3.0*a**2*c**2 - b**2*c**2)/(b**2*c**2) + Ke(11,21)
     1*(a**2*b**2 - 3.0*a**2*c**2)/(b**2*c**2) + Ke(11,24)*(a**2*b**2 
     1- 3.0*a**2*c**2 + b**2*c**2)/(b**2*c**2) + Ke(15,20)*(-a**2*b**2 
     1+ a**2*c**2)/(b**2*c**2) + Ke(15,23)*(a**2*b**2 - 3.0*a**2*c**2)
     1/(b**2*c**2) + Ke(18,20)*(-a**2*b**2 + a**2*c**2 + b**2*c**2)
     1/(b**2*c**2) + Ke(18,23)*(a**2*b**2 - 3.0*a**2*c**2 - b**2*c**2)
     1/(b**2*c**2) + Ke(20,21)*(-a**2*b**2 + a**2*c**2)/(b**2*c**2) 
     1+ Ke(20,24)*(-a**2*b**2 + a**2*c**2 - b**2*c**2)/(b**2*c**2) 
     1+ Ke(21,23)*(a**2*b**2 - 3.0*a**2*c**2)/(b**2*c**2) + Ke(23,24)
     1*(a**2*b**2 - 3.0*a**2*c**2 + b**2*c**2)/(b**2*c**2) + Ke(19,19)
     1*(-4.0*a**2*b**2 + 4.0*a**2*c**2)/(b**3*c) 
     1+ Ke(19,22)*(-8.0*a**2*b**2 + 8.0*a**2*c**2)/(b**3*c) + Ke(22,22)
     1*(-4.0*a**2*b**2 + 4.0*a**2*c**2)/(b**3*c) 
     1+ Ke(15,19)*(-4.0*a**3*b**2 + 4.0*a**3*c**2)/(b**3*c**2) 
     1+ Ke(15,22)*(-4.0*a**3*b**2 + 4.0*a**3*c**2)/(b**3*c**2) 
     1+ Ke(18,19)*(-4.0*a**3*b**2 + 4.0*a**3*c**2 + 2.0*a*b**2*c**2)
     1/(b**3*c**2) + Ke(18,22)*(-4.0*a**3*b**2 + 4.0*a**3*c**2 
     1+ 2.0*a*b**2*c**2)/(b**3*c**2) + Ke(19,21)*(-4.0*a**3*b**2 
     1+ 4.0*a**3*c**2)/(b**3*c**2) + Ke(19,24)*(-4.0*a**3*b**2 
     1 + 4.0*a**3*c**2 - 2.0*a*b**2*c**2)/(b**3*c**2) + Ke(21,22)
     1*(-4.0*a**3*b**2 + 4.0*a**3*c**2)/(b**3*c**2) + Ke(22,24)
     1*(-4.0*a**3*b**2 + 4.0*a**3*c**2 - 2.0*a*b**2*c**2)/(b**3*c**2) 
     1+ Ke(15,15)*(-a**4*b**2 + a**4*c**2)/(b**3*c**3) + Ke(15,18)
     1*(-2.0*a**4*b**2 + 2.0*a**4*c**2 + a**2*b**2*c**2)/(b**3*c**3) 
     1+ Ke(15,21)*(-2.0*a**4*b**2 + 2.0*a**4*c**2)/(b**3*c**3) 
     1+ Ke(15,24)*(-2.0*a**4*b**2 + 2.0*a**4*c**2 - a**2*b**2*c**2)
     1/(b**3*c**3) + Ke(18,18)*(-a**4*b**2 + a**4*c**2 + a**2*b**2*c**2)
     1/(b**3*c**3) + Ke(18,21)*(-2.0*a**4*b**2 + 2.0*a**4*c**2 
     1+ a**2*b**2*c**2)/(b**3*c**3) + Ke(18,24)*(-2.0*a**4*b**2 
     1+ 2.0*a**4*c**2)/(b**3*c**3) + Ke(21,21)*(-a**4*b**2 + a**4*c**2)
     1/(b**3*c**3) + Ke(21,24)*(-2.0*a**4*b**2 + 2.0*a**4*c**2 
     1- a**2*b**2*c**2)/(b**3*c**3) + Ke(24,24)*(-a**4*b**2 + a**4*c**2 
     1- a**2*b**2*c**2)/(b**3*c**3)
      Ke(6,9) = -2.0*a**2*Ke(11,15)/(b*c) - 2.0*a**2*Ke(11,18)/(b*c) 
     1- 2.0*a**2*Ke(15,23)/(b*c) - 2.0*a**2*Ke(18,23)/(b*c) 
     1- 4.0*a*Ke(11,19)/b - 4.0*a*Ke(11,22)/b - 4.0*a*Ke(19,23)/b 
     1- 4.0*a*Ke(22,23)/b - Ke(12,12) + (-8.0*E*a*b**2 - 8.0*E*a*c**2*v 
     1+ 8.0*E*a*c**2)/(3.0*b*c*v**2 - 3.0*b*c) - 2.0*c*Ke(11,12)/b 
     1 - 2.0*c*Ke(12,23)/b + Ke(11,21)*(-2.0*a**2 - 2.0*c**2)/(b*c) 
     1 + Ke(11,24)*(-2.0*a**2 - 2.0*c**2)/(b*c) + Ke(21,23)*(-2.0*a**2 
     1 - 2.0*c**2)/(b*c) + Ke(23,24)*(-2.0*a**2 - 2.0*c**2)/(b*c) 
     1+ Ke(12,19)*(-4.0*a*b**2 + 2.0*a*c**2)/(b**2*c) 
     1+ Ke(12,22)*(-4.0*a*b**2 + 2.0*a*c**2)/(b**2*c) 
     1+ Ke(12,15)*(-2.0*a**2*b**2 + a**2*c**2)/(b**2*c**2) 
     1+ Ke(12,18)*(-2.0*a**2*b**2 + a**2*c**2 + b**2*c**2)/(b**2*c**2) 
     1+ Ke(12,21)*(-2.0*a**2*b**2 + a**2*c**2 - b**2*c**2)/(b**2*c**2) 
     1+ Ke(12,24)*(-2.0*a**2*b**2 + a**2*c**2-2.0*b**2*c**2)/(b**2*c**2)
     1+ Ke(19,19)*(-4.0*a**2*b**2 + 4.0*a**2*c**2)/(b**2*c**2) 
     1+ Ke(19,22)*(-8.0*a**2*b**2 + 8.0*a**2*c**2)/(b**2*c**2) 
     1+ Ke(22,22)*(-4.0*a**2*b**2 + 4.0*a**2*c**2)/(b**2*c**2) 
     1+ Ke(15,19)*(-4.0*a**3*b**2 + 4.0*a**3*c**2)/(b**2*c**3) 
     1+ Ke(15,22)*(-4.0*a**3*b**2 + 4.0*a**3*c**2)/(b**2*c**3) 
     1+ Ke(18,19)*(-4.0*a**3*b**2 + 4.0*a**3*c**2 + 2.0*a*b**2*c**2)
     1/(b**2*c**3) + Ke(18,22)*(-4.0*a**3*b**2 + 4.0*a**3*c**2 
     1+ 2.0*a*b**2*c**2)/(b**2*c**3) + Ke(19,21)*(-4.0*a**3*b**2 
     1+ 4.0*a**3*c**2 - 2.0*a*b**2*c**2 + 2.0*a*c**4)/(b**2*c**3) 
     1+ Ke(19,24)*(-4.0*a**3*b**2 + 4.0*a**3*c**2 - 4.0*a*b**2*c**2 
     1+ 2.0*a*c**4)/(b**2*c**3) + Ke(21,22)*(-4.0*a**3*b**2 
     1+ 4.0*a**3*c**2 - 2.0*a*b**2*c**2 + 2.0*a*c**4)/(b**2*c**3) 
     1+ Ke(22,24)*(-4.0*a**3*b**2 + 4.0*a**3*c**2 - 4.0*a*b**2*c**2 
     1+ 2.0*a*c**4)/(b**2*c**3) + Ke(15,15)*(-a**4*b**2 + a**4*c**2)
     1/(b**2*c**4) + Ke(15,18)*(-2.0*a**4*b**2 + 2.0*a**4*c**2 
     1+ a**2*b**2*c**2)/(b**2*c**4) + Ke(15,21)*(-2.0*a**4*b**2 
     1 + 2.0*a**4*c**2 - a**2*b**2*c**2 + a**2*c**4)/(b**2*c**4) 
     1+ Ke(15,24)*(-2.0*a**4*b**2 + 2.0*a**4*c**2 - 2.0*a**2*b**2*c**2 
     1+ a**2*c**4)/(b**2*c**4) + Ke(18,18)*(-a**4*b**2 + a**4*c**2 
     1+ a**2*b**2*c**2)/(b**2*c**4) + Ke(18,21)*(-2.0*a**4*b**2 
     1+ 2.0*a**4*c**2 + a**2*c**4 + b**2*c**4)/(b**2*c**4) + Ke(18,24)
     1*(-2.0*a**4*b**2 + 2.0*a**4*c**2 - a**2*b**2*c**2 + a**2*c**4 
     1+ b**2*c**4)/(b**2*c**4) + Ke(21,21)*(-a**4*b**2 + a**4*c**2 
     1- a**2*b**2*c**2 + a**2*c**4)/(b**2*c**4) + Ke(21,24)
     1*(-2.0*a**4*b**2 + 2.0*a**4*c**2 - 3.0*a**2*b**2*c**2 
     1+ 2.0*a**2*c**4 - b**2*c**4)/(b**2*c**4) + Ke(24,24)*(-a**4*b**2 
     1+ a**4*c**2 - 2.0*a**2*b**2*c**2 + a**2*c**4 
     1- b**2*c**4)/(b**2*c**4)
      Ke(6,10) = -a*Ke(12,15)/c - a*Ke(12,24)/c - 2.0*a*Ke(11,15)/b 
     1- 2.0*a*Ke(11,24)/b - a*Ke(12,20)/b - a*Ke(12,23)/b 
     1- 2.0*a*c*Ke(11,20)/b**2 - 2.0*a*c*Ke(11,23)/b**2 
     1- 2.0*a*c*Ke(20,23)/b**2 - 2.0*a*c*Ke(23,23)/b**2 - Ke(12,22) 
     1+ (-2.0*E*b**2 - 2.0*E*c**2*v + 2.0*E*c**2)/(3.0*b*v**2 - 3.0*b)
     1- 2.0*c*Ke(11,22)/b + Ke(19,22)*(-2.0*a*b**2 +2.0*a*c**2)/(b**2*c)
     1+ Ke(22,22)*(-2.0*a*b**2 + 2.0*a*c**2)/(b**2*c) 
     1+ Ke(15,19)*(-2.0*a**2*b**2 + 2.0*a**2*c**2)/(b**2*c**2) 
     1+ Ke(15,22)*(-3.0*a**2*b**2 + 3.0*a**2*c**2)/(b**2*c**2) 
     1+ Ke(18,22)*(-a**2*b**2 + a**2*c**2 + b**2*c**2)/(b**2*c**2) 
     1+ Ke(19,24)*(-2.0*a**2*b**2 + 2.0*a**2*c**2)/(b**2*c**2) 
     1+ Ke(21,22)*(-a**2*b**2 + a**2*c**2)/(b**2*c**2) 
     1+ Ke(22,24)*(-3.0*a**2*b**2 + 3.0*a**2*c**2-b**2*c**2)/(b**2*c**2)
     1+ Ke(15,15)*(-a**3*b**2 + a**3*c**2)/(b**2*c**3) 
     1+ Ke(15,18)*(-a**3*b**2 + a**3*c**2 + a*b**2*c**2)/(b**2*c**3) 
     1+ Ke(15,21)*(-a**3*b**2 + a**3*c**2)/(b**2*c**3) 
     1+ Ke(15,24)*(-2.0*a**3*b**2 + 2.0*a**3*c**2 - a*b**2*c**2)
     1/(b**2*c**3) + Ke(18,24)*(-a**3*b**2 + a**3*c**2 + a*b**2*c**2)
     1/(b**2*c**3) + Ke(21,24)*(-a**3*b**2 + a**3*c**2)/(b**2*c**3) 
     1+ Ke(24,24)*(-a**3*b**2 + a**3*c**2 - a*b**2*c**2)/(b**2*c**3) 
     1+ Ke(19,20)*(-2.0*a**2*b**2 + 2.0*a**2*c**2)/(b**3*c) 
     1+ Ke(19,23)*(-2.0*a**2*b**2 + 2.0*a**2*c**2)/(b**3*c) 
     1+ Ke(20,22)*(-2.0*a**2*b**2 + 2.0*a**2*c**2)/(b**3*c) 
     1+ Ke(22,23)*(-2.0*a**2*b**2 + 2.0*a**2*c**2 - 2.0*b**2*c**2)
     1/(b**3*c) + Ke(15,20)*(-a**3*b**2 + a**3*c**2)/(b**3*c**2) 
     1+ Ke(15,23)*(-a**3*b**2 + a**3*c**2 - 2.0*a*b**2*c**2)
     1/(b**3*c**2) + Ke(18,20)*(-a**3*b**2 + a**3*c**2 + a*b**2*c**2)
     1/(b**3*c**2) + Ke(18,23)*(-a**3*b**2 + a**3*c**2 + a*b**2*c**2)
     1/(b**3*c**2) + Ke(20,21)*(-a**3*b**2 + a**3*c**2)/(b**3*c**2) 
     1+ Ke(20,24)*(-a**3*b**2 + a**3*c**2 - a*b**2*c**2)/(b**3*c**2) 
     1+ Ke(21,23)*(-a**3*b**2 + a**3*c**2)/(b**3*c**2) 
     1+ Ke(23,24)*(-a**3*b**2 + a**3*c**2 - 3.0*a*b**2*c**2)/(b**3*c**2)
      Ke(6,11) = Ke(11,12) + (2.0*E*a*b**2*v - 2.0*E*a*c**2*v 
     1 + 2.0*E*a*c**2)/(3.0*b**2*v**2 - 3.0*b**2) + 2.0*c*Ke(11,11)/b 
     1+ 2.0*c*Ke(11,23)/b + Ke(11,19)*(2.0*a*b**2 - 2.0*a*c**2)/(b**2*c)
     1+ Ke(11,22)*(2.0*a*b**2 - 2.0*a*c**2)/(b**2*c) 
     1+ Ke(11,15)*(a**2*b**2 - a**2*c**2)/(b**2*c**2) 
     1+ Ke(11,18)*(a**2*b**2 - a**2*c**2 - b**2*c**2)/(b**2*c**2) 
     1+ Ke(11,21)*(a**2*b**2 - a**2*c**2)/(b**2*c**2) 
     1+ Ke(11,24)*(a**2*b**2 - a**2*c**2 + b**2*c**2)/(b**2*c**2)
      Ke(6,12) = Ke(12,12) + 2.0*c*Ke(11,12)/b + 2.0*c*Ke(12,23)/b 
     1+ Ke(12,19)*(2.0*a*b**2 - 2.0*a*c**2)/(b**2*c) 
     1+ Ke(12,22)*(2.0*a*b**2 - 2.0*a*c**2)/(b**2*c) 
     1+ Ke(12,15)*(a**2*b**2 - a**2*c**2)/(b**2*c**2) 
     1+ Ke(12,18)*(a**2*b**2 - a**2*c**2 - b**2*c**2)/(b**2*c**2) 
     1+ Ke(12,21)*(a**2*b**2 - a**2*c**2)/(b**2*c**2) 
     1+ Ke(12,24)*(a**2*b**2 - a**2*c**2 + b**2*c**2)/(b**2*c**2)
      Ke(6,13) = a*Ke(12,15)/c + a*Ke(12,18)/c + a*Ke(12,21)/c 
     1+ a*Ke(12,24)/c + 2.0*a*Ke(11,15)/b + 2.0*a*Ke(11,18)/b 
     1 + 2.0*a*Ke(11,21)/b + 2.0*a*Ke(11,24)/b + 2.0*a*Ke(15,23)/b 
     1+ 2.0*a*Ke(18,23)/b + 2.0*a*Ke(21,23)/b + 2.0*a*Ke(23,24)/b 
     1- Ke(12,16) + Ke(12,19) + Ke(12,22) + (E*b**2 + E*c**2*v 
     1- E*c**2)/(b*v**2 - b) - 2.0*c*Ke(11,16)/b + 2.0*c*Ke(11,19)/b 
     1+ 2.0*c*Ke(11,22)/b - 2.0*c*Ke(16,23)/b + 2.0*c*Ke(19,23)/b 
     1+ 2.0*c*Ke(22,23)/b + Ke(16,19)*(-2.0*a*b**2 +2.0*a*c**2)/(b**2*c)
     1+ Ke(16,22)*(-2.0*a*b**2 + 2.0*a*c**2)/(b**2*c) 
     1+ Ke(19,19)*(2.0*a*b**2 - 2.0*a*c**2)/(b**2*c) 
     1+ Ke(19,22)*(4.0*a*b**2 - 4.0*a*c**2)/(b**2*c) 
     1+ Ke(22,22)*(2.0*a*b**2 - 2.0*a*c**2)/(b**2*c) 
     1+ Ke(15,16)*(-a**2*b**2 + a**2*c**2)/(b**2*c**2) 
     1+ Ke(15,19)*(3.0*a**2*b**2 - 3.0*a**2*c**2)/(b**2*c**2) 
     1+ Ke(15,22)*(3.0*a**2*b**2 - 3.0*a**2*c**2)/(b**2*c**2) 
     1+ Ke(16,18)*(-a**2*b**2 + a**2*c**2 + b**2*c**2)/(b**2*c**2) 
     1+ Ke(16,21)*(-a**2*b**2 + a**2*c**2)/(b**2*c**2) 
     1+ Ke(16,24)*(-a**2*b**2 + a**2*c**2 - b**2*c**2)/(b**2*c**2) 
     1+ Ke(18,19)*(3.0*a**2*b**2 - 3.0*a**2*c**2-b**2*c**2)/(b**2*c**2) 
     1+ Ke(18,22)*(3.0*a**2*b**2 - 3.0*a**2*c**2 -b**2*c**2)/(b**2*c**2)
     1+ Ke(19,21)*(3.0*a**2*b**2 - 3.0*a**2*c**2)/(b**2*c**2) 
     1+ Ke(19,24)*(3.0*a**2*b**2 - 3.0*a**2*c**2 +b**2*c**2)/(b**2*c**2)
     1+ Ke(21,22)*(3.0*a**2*b**2 - 3.0*a**2*c**2)/(b**2*c**2) 
     1+ Ke(22,24)*(3.0*a**2*b**2 - 3.0*a**2*c**2 +b**2*c**2)/(b**2*c**2)
     1+ Ke(15,15)*(a**3*b**2 - a**3*c**2)/(b**2*c**3) 
     1+ Ke(15,18)*(2.0*a**3*b**2 - 2.0*a**3*c**2 - a*b**2*c**2)
     1/(b**2*c**3) + Ke(15,21)*(2.0*a**3*b**2 - 2.0*a**3*c**2)
     1/(b**2*c**3) + Ke(15,24)*(2.0*a**3*b**2 - 2.0*a**3*c**2 
     1+ a*b**2*c**2)/(b**2*c**3) + Ke(18,18)*(a**3*b**2 - a**3*c**2
     1- a*b**2*c**2)/(b**2*c**3) + Ke(18,21)*(2.0*a**3*b**2 
     1- 2.0*a**3*c**2 - a*b**2*c**2)/(b**2*c**3) 
     1+ Ke(18,24)*(2.0*a**3*b**2 - 2.0*a**3*c**2)/(b**2*c**3) 
     1+ Ke(21,21)*(a**3*b**2 - a**3*c**2)/(b**2*c**3) 
     1+ Ke(21,24)*(2.0*a**3*b**2 - 2.0*a**3*c**2 
     1+ a*b**2*c**2)/(b**2*c**3) + Ke(24,24)*(a**3*b**2 
     1- a**3*c**2 + a*b**2*c**2)/(b**2*c**3)
      Ke(6,14) =-Ke(12,17) - Ke(12,20) - Ke(12,23) + (E*a*b**2*v 
     1- E*a*c**2*v + E*a*c**2)/(3.0*b**2*v**2 - 3.0*b**2) 
     1- 2.0*c*Ke(11,17)/b - 2.0*c*Ke(11,20)/b - 2.0*c*Ke(11,23)/b 
     1- 2.0*c*Ke(17,23)/b - 2.0*c*Ke(20,23)/b - 2.0*c*Ke(23,23)/b 
     1+ Ke(17,19)*(-2.0*a*b**2 + 2.0*a*c**2)/(b**2*c) + Ke(17,22)
     1*(-2.0*a*b**2 + 2.0*a*c**2)/(b**2*c) + Ke(19,20)*(-2.0*a*b**2
     1+ 2.0*a*c**2)/(b**2*c) + Ke(19,23)*(-2.0*a*b**2 + 2.0*a*c**2)
     1/(b**2*c) + Ke(20,22)*(-2.0*a*b**2 + 2.0*a*c**2)/(b**2*c) 
     1+ Ke(22,23)*(-2.0*a*b**2 + 2.0*a*c**2)/(b**2*c) + Ke(15,17)
     1*(-a**2*b**2 + a**2*c**2)/(b**2*c**2) + Ke(15,20)*(-a**2*b**2
     1+ a**2*c**2)/(b**2*c**2) + Ke(15,23)*(-a**2*b**2 + a**2*c**2)
     1/(b**2*c**2) + Ke(17,18)*(-a**2*b**2 + a**2*c**2 + b**2*c**2)
     1/(b**2*c**2) + Ke(17,21)*(-a**2*b**2 + a**2*c**2)/(b**2*c**2)
     1+ Ke(17,24)*(-a**2*b**2 + a**2*c**2 - b**2*c**2)/(b**2*c**2) 
     1+ Ke(18,20)*(-a**2*b**2 + a**2*c**2 + b**2*c**2)/(b**2*c**2) 
     1+ Ke(18,23)*(-a**2*b**2 + a**2*c**2 + b**2*c**2)/(b**2*c**2) 
     1+ Ke(20,21)*(-a**2*b**2 + a**2*c**2)/(b**2*c**2) + Ke(20,24)
     1*(-a**2*b**2 + a**2*c**2 - b**2*c**2)/(b**2*c**2) + Ke(21,23)
     1*(-a**2*b**2 + a**2*c**2)/(b**2*c**2) + Ke(23,24)*(-a**2*b**2
     1+ a**2*c**2 - b**2*c**2)/(b**2*c**2)
      Ke(6,15) = Ke(12,15) + 2.0*c*Ke(11,15)/b + 2.0*c*Ke(15,23)/b 
     1+ Ke(15,19)*(2.0*a*b**2 - 2.0*a*c**2)/(b**2*c) + Ke(15,22)
     1*(2.0*a*b**2 - 2.0*a*c**2)/(b**2*c) + Ke(15,15)*(a**2*b**2 
     1- a**2*c**2)/(b**2*c**2) + Ke(15,18)*(a**2*b**2 - a**2*c**2 
     1- b**2*c**2)/(b**2*c**2) + Ke(15,21)*(a**2*b**2 - a**2*c**2)
     1/(b**2*c**2) + Ke(15,24)*(a**2*b**2 - a**2*c**2 + b**2*c**2)
     1/(b**2*c**2)
      Ke(6,16) = Ke(12,16) + (E*b**2 + E*c**2*v - E*c**2)/(3.0*b*v**2 
     1- 3.0*b) + 2.0*c*Ke(11,16)/b + 2.0*c*Ke(16,23)/b + Ke(16,19)
     1*(2.0*a*b**2 - 2.0*a*c**2)/(b**2*c) + Ke(16,22)*(2.0*a*b**2 
     1- 2.0*a*c**2)/(b**2*c) + Ke(15,16)*(a**2*b**2 - a**2*c**2)
     1/(b**2*c**2) + Ke(16,18)*(a**2*b**2 - a**2*c**2 - b**2*c**2)
     1/(b**2*c**2) + Ke(16,21)*(a**2*b**2 - a**2*c**2)/(b**2*c**2)
     1+ Ke(16,24)*(a**2*b**2 - a**2*c**2 + b**2*c**2)/(b**2*c**2)
      Ke(6,17) = Ke(12,17) + (-E*a*b**2*v + E*a*c**2*v - E*a*c**2)
     1/(3.0*b**2*v**2 - 3.0*b**2) + 2.0*c*Ke(11,17)/b + 2.0*c*Ke(17,23)
     1/b + Ke(17,19)*(2.0*a*b**2 - 2.0*a*c**2)/(b**2*c) + Ke(17,22)
     1*(2.0*a*b**2 - 2.0*a*c**2)/(b**2*c) + Ke(15,17)*(a**2*b**2 
     1- a**2*c**2)/(b**2*c**2) + Ke(17,18)*(a**2*b**2 - a**2*c**2 
     1- b**2*c**2)/(b**2*c**2) + Ke(17,21)*(a**2*b**2 - a**2*c**2)
     1/(b**2*c**2) + Ke(17,24)*(a**2*b**2 - a**2*c**2 + b**2*c**2)
     1/(b**2*c**2)
      Ke(6,18) = Ke(12,18) + 2.0*c*Ke(11,18)/b + 2.0*c*Ke(18,23)/b 
     1+ Ke(18,19)*(2.0*a*b**2 - 2.0*a*c**2)/(b**2*c) + Ke(18,22)
     1*(2.0*a*b**2 - 2.0*a*c**2)/(b**2*c) + Ke(15,18)*(a**2*b**2 
     1- a**2*c**2)/(b**2*c**2) + Ke(18,18)*(a**2*b**2 - a**2*c**2 
     1- b**2*c**2)/(b**2*c**2) + Ke(18,21)*(a**2*b**2 - a**2*c**2)
     1/(b**2*c**2) + Ke(18,24)*(a**2*b**2 - a**2*c**2 + b**2*c**2)
     1/(b**2*c**2)
      Ke(6,19) = Ke(12,19) + (2.0*E*b**2 + 2.0*E*c**2*v - 2.0*E*c**2)
     1/(3.0*b*v**2 - 3.0*b) + 2.0*c*Ke(11,19)/b + 2.0*c*Ke(19,23)/b 
     1+ Ke(19,19)*(2.0*a*b**2 - 2.0*a*c**2)/(b**2*c) + Ke(19,22)
     1*(2.0*a*b**2 - 2.0*a*c**2)/(b**2*c) + Ke(15,19)*(a**2*b**2 
     1- a**2*c**2)/(b**2*c**2) + Ke(18,19)*(a**2*b**2 - a**2*c**2 
     1- b**2*c**2)/(b**2*c**2) + Ke(19,21)*(a**2*b**2 - a**2*c**2)
     1/(b**2*c**2) + Ke(19,24)*(a**2*b**2 - a**2*c**2 + b**2*c**2)
     1/(b**2*c**2)
      Ke(6,20) = Ke(12,20) + (-2.0*E*a*b**2*v + 2.0*E*a*c**2*v 
     1- 2.0*E*a*c**2)/(3.0*b**2*v**2 - 3.0*b**2) + 2.0*c*Ke(11,20)/b 
     1+ 2.0*c*Ke(20,23)/b + Ke(19,20)*(2.0*a*b**2 - 2.0*a*c**2)/(b**2*c)
     1+ Ke(20,22)*(2.0*a*b**2 - 2.0*a*c**2)/(b**2*c) 
     1+ Ke(15,20)*(a**2*b**2 - a**2*c**2)/(b**2*c**2) 
     1+ Ke(18,20)*(a**2*b**2 - a**2*c**2 - b**2*c**2)/(b**2*c**2) 
     1+ Ke(20,21)*(a**2*b**2 - a**2*c**2)/(b**2*c**2) 
     1+ Ke(20,24)*(a**2*b**2 - a**2*c**2 + b**2*c**2)/(b**2*c**2)
      Ke(6,21) = Ke(12,21) + 2.0*c*Ke(11,21)/b + 2.0*c*Ke(21,23)/b 
     1+ Ke(19,21)*(2.0*a*b**2 - 2.0*a*c**2)/(b**2*c) 
     1+ Ke(21,22)*(2.0*a*b**2 - 2.0*a*c**2)/(b**2*c) 
     1+ Ke(15,21)*(a**2*b**2 - a**2*c**2)/(b**2*c**2) 
     1+ Ke(18,21)*(a**2*b**2 - a**2*c**2 - b**2*c**2)/(b**2*c**2) 
     1+ Ke(21,21)*(a**2*b**2 - a**2*c**2)/(b**2*c**2) 
     1+ Ke(21,24)*(a**2*b**2 - a**2*c**2 + b**2*c**2)/(b**2*c**2)
      Ke(6,22) = Ke(12,22) + (2.0*E*b**2 + 2.0*E*c**2*v - 2.0*E*c**2)
     1/(3.0*b*v**2 - 3.0*b) + 2.0*c*Ke(11,22)/b + 2.0*c*Ke(22,23)/b
     1+ Ke(19,22)*(2.0*a*b**2 - 2.0*a*c**2)/(b**2*c) + Ke(22,22)
     1*(2.0*a*b**2 - 2.0*a*c**2)/(b**2*c) + Ke(15,22)*(a**2*b**2 
     1- a**2*c**2)/(b**2*c**2) + Ke(18,22)*(a**2*b**2 - a**2*c**2
     1- b**2*c**2)/(b**2*c**2) + Ke(21,22)*(a**2*b**2 - a**2*c**2)
     1/(b**2*c**2) + Ke(22,24)*(a**2*b**2 - a**2*c**2 + b**2*c**2)
     1/(b**2*c**2)
      Ke(6,23) = Ke(12,23) + (2.0*E*a*b**2*v - 2.0*E*a*c**2*v 
     1+ 2.0*E*a*c**2)/(3.0*b**2*v**2 - 3.0*b**2) + 2.0*c*Ke(11,23)/b 
     1+ 2.0*c*Ke(23,23)/b + Ke(19,23)*(2.0*a*b**2 - 2.0*a*c**2)/(b**2*c)
     1+ Ke(22,23)*(2.0*a*b**2 - 2.0*a*c**2)/(b**2*c) + Ke(15,23)
     1*(a**2*b**2 - a**2*c**2)/(b**2*c**2) + Ke(18,23)*(a**2*b**2 
     1- a**2*c**2 - b**2*c**2)/(b**2*c**2) + Ke(21,23)*(a**2*b**2 
     1- a**2*c**2)/(b**2*c**2) + Ke(23,24)*(a**2*b**2 - a**2*c**2 
     1+ b**2*c**2)/(b**2*c**2)
      Ke(6,24) = Ke(12,24) + 2.0*c*Ke(11,24)/b + 2.0*c*Ke(23,24)/b 
     1+ Ke(19,24)*(2.0*a*b**2 - 2.0*a*c**2)/(b**2*c) + Ke(22,24)
     1*(2.0*a*b**2 - 2.0*a*c**2)/(b**2*c) + Ke(15,24)*(a**2*b**2 
     1- a**2*c**2)/(b**2*c**2) + Ke(18,24)*(a**2*b**2 - a**2*c**2 
     1- b**2*c**2)/(b**2*c**2) + Ke(21,24)*(a**2*b**2 - a**2*c**2)
     1/(b**2*c**2) + Ke(24,24)*(a**2*b**2 
     1- a**2*c**2 + b**2*c**2)/(b**2*c**2)
      Ke(7,7) = -2.0*E*a*c/(3.0*b*v + 3.0*b) + a**2*Ke(18,18)/c**2 
     1+ 2.0*a**2*Ke(18,21)/c**2 + a**2*Ke(21,21)/c**2 
     1- 2.0*a**2*Ke(18,20)/(b*c) - 2.0*a**2*Ke(18,23)/(b*c) 
     1- 2.0*a**2*Ke(20,21)/(b*c) - 2.0*a**2*Ke(21,23)/(b*c) 
     1+ a**2*Ke(20,20)/b**2 + 2.0*a**2*Ke(20,23)/b**2 + a**2*Ke(23,23)
     1/b**2 + 2.0*a*Ke(18,19)/c + 2.0*a*Ke(19,21)/c - 2.0*a*Ke(19,20)/b 
     1- 2.0*a*Ke(19,23)/b + Ke(19,19)
      Ke(7,8) = -E*c/(3.0*v + 3.0) + a**3*Ke(15,18)/(b*c**2) 
     1+ a**3*Ke(15,21)/(b*c**2) + a**3*Ke(18,18)/(b*c**2) 
     1+ 2.0*a**3*Ke(18,21)/(b*c**2) + a**3*Ke(18,24)/(b*c**2) 
     1+ a**3*Ke(21,21)/(b*c**2) + a**3*Ke(21,24)/(b*c**2) 
     1- a**3*Ke(15,20)/(b**2*c) - a**3*Ke(15,23)/(b**2*c) 
     1- a**3*Ke(20,24)/(b**2*c) - a**3*Ke(23,24)/(b**2*c) 
     1+ a**2*Ke(15,19)/(b*c) + 3.0*a**2*Ke(18,19)/(b*c) 
     1+ 2.0*a**2*Ke(18,22)/(b*c) + 3.0*a**2*Ke(19,21)/(b*c) 
     1 + a**2*Ke(19,24)/(b*c) + 2.0*a**2*Ke(21,22)/(b*c) 
     1- 2.0*a**2*Ke(20,22)/b**2 - 2.0*a**2*Ke(22,23)/b**2 
     1- a*Ke(11,18)/c - a*Ke(11,21)/c + a*Ke(11,20)/b + a*Ke(11,23)/b 
     1+ 2.0*a*Ke(19,19)/b + 2.0*a*Ke(19,22)/b - a*Ke(20,20)/b 
     1+ a*Ke(23,23)/b - Ke(11,19) + Ke(19,20)*(-2.0*a**2 + b**2)/b**2 
     1+ Ke(19,23)*(-2.0*a**2 - b**2)/b**2 + Ke(18,20)*(-a**3 + a*b**2)
     1/(b**2*c) + Ke(18,23)*(-a**3 - a*b**2)/(b**2*c) + Ke(20,21)*(-a**3
     1+ a*b**2)/(b**2*c) + Ke(21,23)*(-a**3 - a*b**2)/(b**2*c)
      Ke(7,9) = 2.0*E*b/(3.0*v**2 - 3.0) + a**3*Ke(15,18)/c**3 
     1+ a**3*Ke(15,21)/c**3 + a**3*Ke(18,18)/c**3 
     1- a**3*Ke(15,20)/(b*c**2) - a**3*Ke(15,23)/(b*c**2) 
     1- a**3*Ke(18,20)/(b*c**2) - a**3*Ke(18,23)/(b*c**2) 
     1+ a**2*Ke(15,19)/c**2 + 3.0*a**2*Ke(18,19)/c**2 
     1+ 2.0*a**2*Ke(18,22)/c**2 + 2.0*a**2*Ke(21,22)/c**2 
     1- 2.0*a**2*Ke(19,20)/(b*c) - 2.0*a**2*Ke(19,23)/(b*c) 
     1- 2.0*a**2*Ke(20,22)/(b*c) - 2.0*a**2*Ke(22,23)/(b*c) 
     1+ a*Ke(12,18)/c + a*Ke(12,21)/c + 2.0*a*Ke(19,19)/c 
     1+ 2.0*a*Ke(19,22)/c - a*Ke(12,20)/b - a*Ke(12,23)/b 
     1+ Ke(12,19) + Ke(19,21)*(3.0*a**2 + c**2)/c**2 + Ke(19,24)*(a**2 
     1+ c**2)/c**2 + Ke(18,21)*(2.0*a**3 + a*c**2)/c**3 
     1+ Ke(18,24)*(a**3 + a*c**2)/c**3 + Ke(21,21)*(a**3 + a*c**2)/c**3 
     1+ Ke(21,24)*(a**3 + a*c**2)/c**3 + Ke(20,21)*(-a**3 - a*c**2)
     1/(b*c**2) + Ke(20,24)*(-a**3 - a*c**2)/(b*c**2) + Ke(21,23)*(-a**3
     1- a*c**2)/(b*c**2) + Ke(23,24)*(-a**3 - a*c**2)/(b*c**2)
      Ke(7,10) = 2.0*E*a*c/(3.0*b*v + 3.0*b) + a**2*Ke(15,18)/c**2 
     1+ a**2*Ke(15,21)/c**2 + a**2*Ke(18,24)/c**2 + a**2*Ke(21,24)/c**2 
     1- a**2*Ke(15,20)/(b*c) - a**2*Ke(15,23)/(b*c) 
     1+ a**2*Ke(18,20)/(b*c) + a**2*Ke(18,23)/(b*c) 
     1+ a**2*Ke(20,21)/(b*c) - a**2*Ke(20,24)/(b*c) 
     1+ a**2*Ke(21,23)/(b*c) - a**2*Ke(23,24)/(b*c) -a**2*Ke(20,20)/b**2
     1- 2.0*a**2*Ke(20,23)/b**2 - a**2*Ke(23,23)/b**2 + a*Ke(15,19)/c 
     1+ a*Ke(18,22)/c + a*Ke(19,24)/c + a*Ke(21,22)/c + a*Ke(19,20)/b 
     1+ a*Ke(19,23)/b - a*Ke(20,22)/b - a*Ke(22,23)/b + Ke(19,22)
      Ke(7,11) = E*c/(3.0*v + 3.0) - a*Ke(11,18)/c - a*Ke(11,21)/c 
     1+ a*Ke(11,20)/b + a*Ke(11,23)/b - Ke(11,19)
      Ke(7,12) = -a*Ke(12,18)/c - a*Ke(12,21)/c + a*Ke(12,20)/b 
     1+ a*Ke(12,23)/b - Ke(12,19)
      Ke(7,13) = -a**2*Ke(15,18)/c**2 - a**2*Ke(15,21)/c**2 
     1- a**2*Ke(18,18)/c**2 - 2.0*a**2*Ke(18,21)/c**2 
     1- a**2*Ke(18,24)/c**2 - a**2*Ke(21,21)/c**2 - a**2*Ke(21,24)/c**2 
     1+ a**2*Ke(15,20)/(b*c) + a**2*Ke(15,23)/(b*c)+a**2*Ke(18,20)/(b*c)
     1+ a**2*Ke(18,23)/(b*c) + a**2*Ke(20,21)/(b*c)+a**2*Ke(20,24)/(b*c)
     1+ a**2*Ke(21,23)/(b*c) + a**2*Ke(23,24)/(b*c) - a*Ke(15,19)/c 
     1+ a*Ke(16,18)/c + a*Ke(16,21)/c - 2.0*a*Ke(18,19)/c -a*Ke(18,22)/c
     1- 2.0*a*Ke(19,21)/c - a*Ke(19,24)/c - a*Ke(21,22)/c -a*Ke(16,20)/b
     1- a*Ke(16,23)/b + a*Ke(19,20)/b + a*Ke(19,23)/b + a*Ke(20,22)/b 
     1+ a*Ke(22,23)/b + Ke(16,19) - Ke(19,19) - Ke(19,22) +(-E*a**2*c*v 
     1+ E*a**2*c - E*b**2*c)/(6.0*a*b*v**2 - 6.0*a*b)
      Ke(7,14) = 7.0*E*c/(12.0*v + 12.0) + a*Ke(17,18)/c + a*Ke(17,21)/c
     1+ a*Ke(18,20)/c + a*Ke(18,23)/c + a*Ke(20,21)/c + a*Ke(21,23)/c 
     1- a*Ke(17,20)/b - a*Ke(17,23)/b - a*Ke(20,20)/b-2.0*a*Ke(20,23)/b
     1- a*Ke(23,23)/b + Ke(17,19) + Ke(19,20) + Ke(19,23)
      Ke(7,15) = -a*Ke(15,18)/c - a*Ke(15,21)/c + a*Ke(15,20)/b 
     1+ a*Ke(15,23)/b - Ke(15,19)
      Ke(7,16) = E*a*c/(6.0*b*v + 6.0*b) - a*Ke(16,18)/c - a*Ke(16,21)/c
     1+ a*Ke(16,20)/b + a*Ke(16,23)/b - Ke(16,19)
      Ke(7,17) = -E*c/(6.0*v + 6.0) - a*Ke(17,18)/c - a*Ke(17,21)/c 
     1+ a*Ke(17,20)/b + a*Ke(17,23)/b - Ke(17,19)
      Ke(7,18) = -a*Ke(18,18)/c - a*Ke(18,21)/c + a*Ke(18,20)/b 
     1+ a*Ke(18,23)/b - Ke(18,19)
      Ke(7,19) = E*a*c/(3.0*b*v + 3.0*b) - a*Ke(18,19)/c - a*Ke(19,21)/c
     1+ a*Ke(19,20)/b + a*Ke(19,23)/b - Ke(19,19)
      Ke(7,20) = -E*c/(3.0*v + 3.0) - a*Ke(18,20)/c - a*Ke(20,21)/c 
     1+ a*Ke(20,20)/b + a*Ke(20,23)/b - Ke(19,20)
      Ke(7,21) = -a*Ke(18,21)/c - a*Ke(21,21)/c + a*Ke(20,21)/b 
     1+ a*Ke(21,23)/b - Ke(19,21)
      Ke(7,22) = -E*a*c/(3.0*b*v + 3.0*b) - a*Ke(18,22)/c-a*Ke(21,22)/c
     1+ a*Ke(20,22)/b + a*Ke(22,23)/b - Ke(19,22)
      Ke(7,23) = -E*c/(3.0*v + 3.0) - a*Ke(18,23)/c - a*Ke(21,23)/c 
     1+ a*Ke(20,23)/b + a*Ke(23,23)/b - Ke(19,23)
      Ke(7,24) = -a*Ke(18,24)/c - a*Ke(21,24)/c + a*Ke(20,24)/b 
     1+ a*Ke(23,24)/b - Ke(19,24)
      Ke(8,8) = -4.0*E*a*c/(b*v + b) + a**4*Ke(15,15)/(b**2*c**2) 
     1+ 2.0*a**4*Ke(15,18)/(b**2*c**2) + 2.0*a**4*Ke(15,21)/(b**2*c**2) 
     1+ 2.0*a**4*Ke(15,24)/(b**2*c**2) + a**4*Ke(18,18)/(b**2*c**2) 
     1+ 2.0*a**4*Ke(18,21)/(b**2*c**2) + 2.0*a**4*Ke(18,24)/(b**2*c**2) 
     1+ a**4*Ke(21,21)/(b**2*c**2) + 2.0*a**4*Ke(21,24)/(b**2*c**2) 
     1+ a**4*Ke(24,24)/(b**2*c**2) + 4.0*a**3*Ke(15,19)/(b**2*c) 
     1+ 4.0*a**3*Ke(15,22)/(b**2*c) + 4.0*a**3*Ke(18,19)/(b**2*c) 
     1+ 4.0*a**3*Ke(18,22)/(b**2*c) + 4.0*a**3*Ke(19,21)/(b**2*c) 
     1+ 4.0*a**3*Ke(19,24)/(b**2*c) + 4.0*a**3*Ke(21,22)/(b**2*c) 
     1+ 4.0*a**3*Ke(22,24)/(b**2*c) - 2.0*a**2*Ke(11,15)/(b*c) 
     1- 2.0*a**2*Ke(11,18)/(b*c) - 2.0*a**2*Ke(11,21)/(b*c) 
     1- 2.0*a**2*Ke(11,24)/(b*c) + 2.0*a**2*Ke(15,20)/(b*c) 
     1- 2.0*a**2*Ke(15,23)/(b*c) + 2.0*a**2*Ke(18,20)/(b*c) 
     1- 2.0*a**2*Ke(18,23)/(b*c) + 2.0*a**2*Ke(20,21)/(b*c) 
     1 + 2.0*a**2*Ke(20,24)/(b*c) - 2.0*a**2*Ke(21,23)/(b*c) 
     1- 2.0*a**2*Ke(23,24)/(b*c) + 4.0*a**2*Ke(19,19)/b**2 
     1+ 8.0*a**2*Ke(19,22)/b**2 + 4.0*a**2*Ke(22,22)/b**2 
     1- 4.0*a*Ke(11,19)/b - 4.0*a*Ke(11,22)/b + 4.0*a*Ke(19,20)/b 
     1- 4.0*a*Ke(19,23)/b + 4.0*a*Ke(20,22)/b - 4.0*a*Ke(22,23)/b 
     1+ Ke(11,11) - 2.0*Ke(11,20) + 2.0*Ke(11,23) 
     1+ Ke(20,20) - 2.0*Ke(20,23) + Ke(23,23)
      Ke(8,9) = a**4*Ke(15,15)/(b*c**3) + 2.0*a**4*Ke(15,18)/(b*c**3) 
     1+ a**4*Ke(18,18)/(b*c**3) + 4.0*a**3*Ke(15,19)/(b*c**2) 
     1+ 4.0*a**3*Ke(15,22)/(b*c**2) + 4.0*a**3*Ke(18,19)/(b*c**2) 
     1+ 4.0*a**3*Ke(18,22)/(b*c**2) - a**2*Ke(11,15)/c**2 
     1- a**2*Ke(11,18)/c**2 + a**2*Ke(15,20)/c**2 - a**2*Ke(15,23)/c**2 
     1+ a**2*Ke(18,20)/c**2 - a**2*Ke(18,23)/c**2 + a**2*Ke(12,15)/(b*c)
     1+ a**2*Ke(12,18)/(b*c) + a**2*Ke(12,21)/(b*c)+a**2*Ke(12,24)/(b*c)
     1+ 4.0*a**2*Ke(19,19)/(b*c) + 8.0*a**2*Ke(19,22)/(b*c) 
     1+ 4.0*a**2*Ke(22,22)/(b*c) - 2.0*a*Ke(11,19)/c - 2.0*a*Ke(11,22)/c
     1+ 2.0*a*Ke(19,20)/c - 2.0*a*Ke(19,23)/c + 2.0*a*Ke(20,22)/c 
     1- 2.0*a*Ke(22,23)/c + 2.0*a*Ke(12,19)/b + 2.0*a*Ke(12,22)/b 
     1- Ke(11,12) + Ke(12,20) - Ke(12,23) + (-6.0*E*a*v + 8.0*E*a)
     1/(3.0*v**2 - 3.0) + Ke(11,21)*(-a**2 - c**2)/c**2 + Ke(11,24)
     1*(-a**2 - c**2)/c**2 + Ke(20,21)*(a**2 + c**2)/c**2 + Ke(20,24)
     1*(a**2 + c**2)/c**2 + Ke(21,23)*(-a**2 - c**2)/c**2 + Ke(23,24)
     1*(-a**2 - c**2)/c**2 + Ke(19,21)*(4.0*a**3 + 2.0*a*c**2)/(b*c**2)
     1+ Ke(19,24)*(4.0*a**3 + 2.0*a*c**2)/(b*c**2) + Ke(21,22)*(4.0*a**3
     1+ 2.0*a*c**2)/(b*c**2) + Ke(22,24)*(4.0*a**3 +2.0*a*c**2)/(b*c**2)
     1+ Ke(15,21)*(2.0*a**4 + a**2*c**2)/(b*c**3) + Ke(15,24)*(2.0*a**4 
     1+ a**2*c**2)/(b*c**3) + Ke(18,21)*(2.0*a**4 + a**2*c**2)/(b*c**3) 
     1+ Ke(18,24)*(2.0*a**4 + a**2*c**2)/(b*c**3) + Ke(21,21)*(a**4 
     1+ a**2*c**2)/(b*c**3) + Ke(21,24)*(2.0*a**4 + 2.0*a**2*c**2)
     1/(b*c**3) + Ke(24,24)*(a**4 + a**2*c**2)/(b*c**3)
      Ke(8,10) = -E*c/(v + 1.0) + a**3*Ke(15,15)/(b*c**2) 
     1+ a**3*Ke(15,18)/(b*c**2) + a**3*Ke(15,21)/(b*c**2) 
     1+ 2.0*a**3*Ke(15,24)/(b*c**2) + a**3*Ke(18,24)/(b*c**2) 
     1 + a**3*Ke(21,24)/(b*c**2) + a**3*Ke(24,24)/(b*c**2) 
     1+ a**3*Ke(18,20)/(b**2*c) + a**3*Ke(18,23)/(b**2*c) 
     1+ a**3*Ke(20,21)/(b**2*c) + a**3*Ke(21,23)/(b**2*c) 
     1 + 2.0*a**2*Ke(15,19)/(b*c) + 3.0*a**2*Ke(15,22)/(b*c) 
     1 + a**2*Ke(18,22)/(b*c) + 2.0*a**2*Ke(19,24)/(b*c) 
     1 + a**2*Ke(21,22)/(b*c) + 3.0*a**2*Ke(22,24)/(b*c) 
     1+ 2.0*a**2*Ke(19,20)/b**2 + 2.0*a**2*Ke(19,23)/b**2 
     1 - a*Ke(11,15)/c - a*Ke(11,24)/c - a*Ke(11,20)/b 
     1- a*Ke(11,23)/b + 2.0*a*Ke(19,22)/b + a*Ke(20,20)/b 
     1+ 2.0*a*Ke(22,22)/b - a*Ke(23,23)/b - Ke(11,22) 
     1+ Ke(20,22)*(2.0*a**2 + b**2)/b**2 + Ke(22,23)*(2.0*a**2 
     1- b**2)/b**2 + Ke(15,20)*(a**3 + a*b**2)/(b**2*c) 
     1+ Ke(15,23)*(a**3 - a*b**2)/(b**2*c) + Ke(20,24)*(a**3 
     1+ a*b**2)/(b**2*c) + Ke(23,24)*(a**3 - a*b**2)/(b**2*c)
      Ke(8,11) = -2.0*E*a*c/(3.0*b*v + 3.0*b) - a**2*Ke(11,15)/(b*c) 
     1- a**2*Ke(11,18)/(b*c) - a**2*Ke(11,21)/(b*c)-a**2*Ke(11,24)/(b*c) 
     1- 2.0*a*Ke(11,19)/b - 2.0*a*Ke(11,22)/b 
     1+ Ke(11,11) - Ke(11,20) + Ke(11,23)
      Ke(8,12) = -a**2*Ke(12,15)/(b*c) - a**2*Ke(12,18)/(b*c) 
     1- a**2*Ke(12,21)/(b*c) - a**2*Ke(12,24)/(b*c) - 2.0*a*Ke(12,19)/b 
     1- 2.0*a*Ke(12,22)/b + Ke(11,12) - Ke(12,20) + Ke(12,23)
      Ke(8,13) = -a**3*Ke(15,15)/(b*c**2) - 2.0*a**3*Ke(15,18)/(b*c**2)
     1- 2.0*a**3*Ke(15,21)/(b*c**2) - 2.0*a**3*Ke(15,24)/(b*c**2) 
     1- a**3*Ke(18,18)/(b*c**2) - 2.0*a**3*Ke(18,21)/(b*c**2) 
     1- 2.0*a**3*Ke(18,24)/(b*c**2) - a**3*Ke(21,21)/(b*c**2) 
     1 - 2.0*a**3*Ke(21,24)/(b*c**2) - a**3*Ke(24,24)/(b*c**2) 
     1+ a**2*Ke(15,16)/(b*c) - 3.0*a**2*Ke(15,19)/(b*c) 
     1- 3.0*a**2*Ke(15,22)/(b*c) + a**2*Ke(16,18)/(b*c)
     1+ a**2*Ke(16,21)/(b*c) + a**2*Ke(16,24)/(b*c) 
     1- 3.0*a**2*Ke(18,19)/(b*c) - 3.0*a**2*Ke(18,22)/(b*c) 
     1- 3.0*a**2*Ke(19,21)/(b*c) - 3.0*a**2*Ke(19,24)/(b*c) 
     1 - 3.0*a**2*Ke(21,22)/(b*c) - 3.0*a**2*Ke(22,24)/(b*c) 
     1 + a*Ke(11,15)/c + a*Ke(11,18)/c + a*Ke(11,21)/c + a*Ke(11,24)/c
     1- a*Ke(15,20)/c + a*Ke(15,23)/c - a*Ke(18,20)/c + a*Ke(18,23)/c 
     1- a*Ke(20,21)/c - a*Ke(20,24)/c + a*Ke(21,23)/c + a*Ke(23,24)/c 
     1+ 2.0*a*Ke(16,19)/b + 2.0*a*Ke(16,22)/b - 2.0*a*Ke(19,19)/b 
     1- 4.0*a*Ke(19,22)/b - 2.0*a*Ke(22,22)/b - Ke(11,16) + Ke(11,19) 
     1+ Ke(11,22) + Ke(16,20) - Ke(16,23) - Ke(19,20) + Ke(19,23) 
     1- Ke(20,22) + Ke(22,23) + (5.0*E*c*v - 6.0*E*c)/(6.0*v**2 - 6.0)
      Ke(8,14) = a**2*Ke(15,17)/(b*c) + a**2*Ke(15,20)/(b*c) 
     1+ a**2*Ke(15,23)/(b*c) + a**2*Ke(17,18)/(b*c) 
     1+ a**2*Ke(17,21)/(b*c) + a**2*Ke(17,24)/(b*c)+a**2*Ke(18,20)/(b*c)
     1+ a**2*Ke(18,23)/(b*c) + a**2*Ke(20,21)/(b*c)+a**2*Ke(20,24)/(b*c)
     1+ a**2*Ke(21,23)/(b*c) + a**2*Ke(23,24)/(b*c) + 2.0*a*Ke(17,19)/b 
     1+ 2.0*a*Ke(17,22)/b + 2.0*a*Ke(19,20)/b + 2.0*a*Ke(19,23)/b 
     1+ 2.0*a*Ke(20,22)/b + 2.0*a*Ke(22,23)/b - Ke(11,17) - Ke(11,20) 
     1- Ke(11,23) + Ke(17,20) - Ke(17,23) + Ke(20,20) - Ke(23,23) 
     1+ (-4.0*E*a**2*c - 3.0*E*b**2*c)/(12.0*a*b*v + 12.0*a*b)
      Ke(8,15) = -a**2*Ke(15,15)/(b*c) - a**2*Ke(15,18)/(b*c) 
     1- a**2*Ke(15,21)/(b*c) - a**2*Ke(15,24)/(b*c) - 2.0*a*Ke(15,19)/b 
     1 - 2.0*a*Ke(15,22)/b + Ke(11,15) - Ke(15,20) + Ke(15,23)
      Ke(8,16) = E*c/(3.0*v + 3.0) - a**2*Ke(15,16)/(b*c) 
     1- a**2*Ke(16,18)/(b*c) - a**2*Ke(16,21)/(b*c)-a**2*Ke(16,24)/(b*c)
     1- 2.0*a*Ke(16,19)/b - 2.0*a*Ke(16,22)/b 
     1+ Ke(11,16) - Ke(16,20) + Ke(16,23)
      Ke(8,17) = E*a*c/(3.0*b*v + 3.0*b) - a**2*Ke(15,17)/(b*c) 
     1- a**2*Ke(17,18)/(b*c) - a**2*Ke(17,21)/(b*c)-a**2*Ke(17,24)/(b*c)
     1- 2.0*a*Ke(17,19)/b - 2.0*a*Ke(17,22)/b + Ke(11,17) 
     1- Ke(17,20) + Ke(17,23)
      Ke(8,18) = -a**2*Ke(15,18)/(b*c) - a**2*Ke(18,18)/(b*c) 
     1- a**2*Ke(18,21)/(b*c) - a**2*Ke(18,24)/(b*c) - 2.0*a*Ke(18,19)/b 
     1- 2.0*a*Ke(18,22)/b + Ke(11,18) - Ke(18,20) + Ke(18,23)
      Ke(8,19) = 2.0*E*c/(3.0*v + 3.0) - a**2*Ke(15,19)/(b*c) 
     1- a**2*Ke(18,19)/(b*c) - a**2*Ke(19,21)/(b*c)-a**2*Ke(19,24)/(b*c)
     1- 2.0*a*Ke(19,19)/b - 2.0*a*Ke(19,22)/b + Ke(11,19) 
     1- Ke(19,20) + Ke(19,23)
      Ke(8,20) = 2.0*E*a*c/(3.0*b*v + 3.0*b) - a**2*Ke(15,20)/(b*c) 
     1- a**2*Ke(18,20)/(b*c) - a**2*Ke(20,21)/(b*c)-a**2*Ke(20,24)/(b*c)
     1- 2.0*a*Ke(19,20)/b - 2.0*a*Ke(20,22)/b 
     1+ Ke(11,20) - Ke(20,20) + Ke(20,23)
      Ke(8,21) = -a**2*Ke(15,21)/(b*c) - a**2*Ke(18,21)/(b*c) 
     1- a**2*Ke(21,21)/(b*c) - a**2*Ke(21,24)/(b*c) - 2.0*a*Ke(19,21)/b 
     1- 2.0*a*Ke(21,22)/b + Ke(11,21) - Ke(20,21) + Ke(21,23)
      Ke(8,22) = 2.0*E*c/(3.0*v + 3.0) - a**2*Ke(15,22)/(b*c) 
     1- a**2*Ke(18,22)/(b*c) - a**2*Ke(21,22)/(b*c) 
     1- a**2*Ke(22,24)/(b*c) - 2.0*a*Ke(19,22)/b - 2.0*a*Ke(22,22)/b 
     1+ Ke(11,22) - Ke(20,22) + Ke(22,23)
      Ke(8,23) = -2.0*E*a*c/(3.0*b*v + 3.0*b) - a**2*Ke(15,23)/(b*c) 
     1- a**2*Ke(18,23)/(b*c) - a**2*Ke(21,23)/(b*c) 
     1- a**2*Ke(23,24)/(b*c) - 2.0*a*Ke(19,23)/b - 2.0*a*Ke(22,23)/b 
     1+ Ke(11,23) - Ke(20,23) + Ke(23,23)
      Ke(8,24) = -a**2*Ke(15,24)/(b*c) - a**2*Ke(18,24)/(b*c) 
     1- a**2*Ke(21,24)/(b*c) - a**2*Ke(24,24)/(b*c) - 2.0*a*Ke(19,24)/b 
     1- 2.0*a*Ke(22,24)/b + Ke(11,24) - Ke(20,24) + Ke(23,24)
      Ke(9,9) = 8.0*E*a*b/(3.0*c*v**2 - 3.0*c) + a**4*Ke(15,15)/c**4 
     1+ 2.0*a**4*Ke(15,18)/c**4 + a**4*Ke(18,18)/c**4 
     1+ 4.0*a**3*Ke(15,19)/c**3 + 4.0*a**3*Ke(15,22)/c**3 
     1+ 4.0*a**3*Ke(18,19)/c**3 + 4.0*a**3*Ke(18,22)/c**3 
     1 + 2.0*a**2*Ke(12,15)/c**2 + 2.0*a**2*Ke(12,18)/c**2 
     1+ 4.0*a**2*Ke(19,19)/c**2 + 8.0*a**2*Ke(19,22)/c**2 
     1+ 4.0*a**2*Ke(22,22)/c**2 + 4.0*a*Ke(12,19)/c + 4.0*a*Ke(12,22)/c 
     1+ Ke(12,12) + Ke(12,21)*(2.0*a**2 + 2.0*c**2)/c**2 
     1+ Ke(12,24)*(2.0*a**2 + 2.0*c**2)/c**2 + Ke(19,21)*(4.0*a**3 
     1+ 4.0*a*c**2)/c**3 + Ke(19,24)*(4.0*a**3 + 4.0*a*c**2)/c**3 
     1+ Ke(21,22)*(4.0*a**3 + 4.0*a*c**2)/c**3 + Ke(22,24)*(4.0*a**3 
     1+ 4.0*a*c**2)/c**3 + Ke(15,21)*(2.0*a**4 + 2.0*a**2*c**2)/c**4 
     1+ Ke(15,24)*(2.0*a**4 + 2.0*a**2*c**2)/c**4 + Ke(18,21)*(2.0*a**4 
     1+ 2.0*a**2*c**2)/c**4 + Ke(18,24)*(2.0*a**4 + 2.0*a**2*c**2)/c**4 
     1+ Ke(21,21)*(a**4 + 2.0*a**2*c**2 + c**4)/c**4 
     1+ Ke(21,24)*(2.0*a**4 + 4.0*a**2*c**2 + 2.0*c**4)/c**4 
     1+ Ke(24,24)*(a**4 + 2.0*a**2*c**2 + c**4)/c**4
      Ke(9,10) = 2.0*E*b/(3.0*v**2 - 3.0) + a**3*Ke(15,15)/c**3 
     1+ a**3*Ke(15,18)/c**3 + a**3*Ke(18,24)/c**3 + a**3*Ke(15,20)
     1/(b*c**2) + a**3*Ke(15,23)/(b*c**2) + a**3*Ke(18,20)/(b*c**2) 
     1+ a**3*Ke(18,23)/(b*c**2) + 2.0*a**2*Ke(15,19)/c**2 
     1+ 3.0*a**2*Ke(15,22)/c**2 + a**2*Ke(18,22)/c**2 
     1+ 2.0*a**2*Ke(19,24)/c**2 + 2.0*a**2*Ke(19,20)/(b*c) 
     1+ 2.0*a**2*Ke(19,23)/(b*c) + 2.0*a**2*Ke(20,22)/(b*c) 
     1+ 2.0*a**2*Ke(22,23)/(b*c) + a*Ke(12,15)/c + a*Ke(12,24)/c 
     1+ 2.0*a*Ke(19,22)/c + 2.0*a*Ke(22,22)/c + a*Ke(12,20)/b 
     1+ a*Ke(12,23)/b + Ke(12,22) + Ke(21,22)*(a**2 + c**2)/c**2 
     1+ Ke(22,24)*(3.0*a**2 + c**2)/c**2 + Ke(15,21)*(a**3 + a*c**2)
     1/c**3 + Ke(15,24)*(2.0*a**3 + a*c**2)/c**3 + Ke(21,24)*(a**3 
     1+ a*c**2)/c**3 + Ke(24,24)*(a**3 + a*c**2)/c**3 
     1+ Ke(20,21)*(a**3 + a*c**2)/(b*c**2) + Ke(20,24)*(a**3 + a*c**2)
     1/(b*c**2) + Ke(21,23)*(a**3 + a*c**2)/(b*c**2) 
     1+ Ke(23,24)*(a**3 + a*c**2)/(b*c**2)
      Ke(9,11) = -2.0*E*a*v/(3.0*v**2 - 3.0) - a**2*Ke(11,15)/c**2 
     1- a**2*Ke(11,18)/c**2 - 2.0*a*Ke(11,19)/c - 2.0*a*Ke(11,22)/c 
     1- Ke(11,12) + Ke(11,21)*(-a**2 - c**2)/c**2 
     1+ Ke(11,24)*(-a**2 - c**2)/c**2
      Ke(9,12) = -a**2*Ke(12,15)/c**2 - a**2*Ke(12,18)/c**2 
     1- 2.0*a*Ke(12,19)/c - 2.0*a*Ke(12,22)/c - Ke(12,12) 
     1+ Ke(12,21)*(-a**2 - c**2)/c**2 + Ke(12,24)*(-a**2 - c**2)/c**2
      Ke(9,13) = -E*b/(v**2 - 1.0) - a**3*Ke(15,15)/c**3 
     1- 2.0*a**3*Ke(15,18)/c**3 - a**3*Ke(18,18)/c**3 
     1+ a**2*Ke(15,16)/c**2 - 3.0*a**2*Ke(15,19)/c**2 
     1- 3.0*a**2*Ke(15,22)/c**2 + a**2*Ke(16,18)/c**2 
     1- 3.0*a**2*Ke(18,19)/c**2 - 3.0*a**2*Ke(18,22)/c**2 
     1- a*Ke(12,15)/c - a*Ke(12,18)/c - a*Ke(12,21)/c - a*Ke(12,24)/c 
     1+ 2.0*a*Ke(16,19)/c + 2.0*a*Ke(16,22)/c - 2.0*a*Ke(19,19)/c 
     1- 4.0*a*Ke(19,22)/c - 2.0*a*Ke(22,22)/c + Ke(12,16) - Ke(12,19) 
     1- Ke(12,22) + Ke(16,21)*(a**2 + c**2)/c**2 + Ke(16,24)*(a**2 
     1+ c**2)/c**2 + Ke(19,21)*(-3.0*a**2 - c**2)/c**2 
     1+ Ke(19,24)*(-3.0*a**2 - c**2)/c**2 + Ke(21,22)*(-3.0*a**2 
     1- c**2)/c**2 + Ke(22,24)*(-3.0*a**2 - c**2)/c**2 
     1+ Ke(15,21)*(-2.0*a**3 - a*c**2)/c**3 + Ke(15,24)*(-2.0*a**3 
     1- a*c**2)/c**3 + Ke(18,21)*(-2.0*a**3 - a*c**2)/c**3 
     1+ Ke(18,24)*(-2.0*a**3 - a*c**2)/c**3 + Ke(21,21)*(-a**3 
     1- a*c**2)/c**3 + Ke(21,24)*(-2.0*a**3 - 2.0*a*c**2)/c**3 
     1+ Ke(24,24)*(-a**3 - a*c**2)/c**3
      Ke(9,14) = -E*a*v/(3.0*v**2 - 3.0) + a**2*Ke(15,17)/c**2 
     1+ a**2*Ke(15,20)/c**2 + a**2*Ke(15,23)/c**2 + a**2*Ke(17,18)/c**2 
     1+ a**2*Ke(18,20)/c**2 + a**2*Ke(18,23)/c**2 + 2.0*a*Ke(17,19)/c 
     1+ 2.0*a*Ke(17,22)/c + 2.0*a*Ke(19,20)/c + 2.0*a*Ke(19,23)/c 
     1+ 2.0*a*Ke(20,22)/c + 2.0*a*Ke(22,23)/c + Ke(12,17) + Ke(12,20) 
     1+ Ke(12,23) + Ke(17,21)*(a**2 + c**2)/c**2 + Ke(17,24)*(a**2 
     1+ c**2)/c**2 + Ke(20,21)*(a**2 + c**2)/c**2 + Ke(20,24)*(a**2 
     1+ c**2)/c**2 + Ke(21,23)*(a**2 + c**2)/c**2 + Ke(23,24)*(a**2 
     1+ c**2)/c**2
      Ke(9,15) = -a**2*Ke(15,15)/c**2 - a**2*Ke(15,18)/c**2 
     1- 2.0*a*Ke(15,19)/c - 2.0*a*Ke(15,22)/c - Ke(12,15) 
     1+ Ke(15,21)*(-a**2 - c**2)/c**2 + Ke(15,24)*(-a**2 - c**2)/c**2
      Ke(9,16) = -E*b/(3.0*v**2 - 3.0) - a**2*Ke(15,16)/c**2 
     1- a**2*Ke(16,18)/c**2 - 2.0*a*Ke(16,19)/c - 2.0*a*Ke(16,22)/c
     1- Ke(12,16) + Ke(16,21)*(-a**2 - c**2)/c**2 
     1+ Ke(16,24)*(-a**2 - c**2)/c**2
      Ke(9,17) = E*a*v/(3.0*v**2 - 3.0) - a**2*Ke(15,17)/c**2 
     1- a**2*Ke(17,18)/c**2 - 2.0*a*Ke(17,19)/c - 2.0*a*Ke(17,22)/c 
     1- Ke(12,17) + Ke(17,21)*(-a**2 - c**2)/c**2 
     1+ Ke(17,24)*(-a**2 - c**2)/c**2
      Ke(9,18) = -a**2*Ke(15,18)/c**2 - a**2*Ke(18,18)/c**2 
     1- 2.0*a*Ke(18,19)/c - 2.0*a*Ke(18,22)/c - Ke(12,18) 
     1+ Ke(18,21)*(-a**2 - c**2)/c**2 + Ke(18,24)*(-a**2 - c**2)/c**2
      Ke(9,19) = -2.0*E*b/(3.0*v**2 - 3.0) - a**2*Ke(15,19)/c**2 
     1- a**2*Ke(18,19)/c**2 - 2.0*a*Ke(19,19)/c - 2.0*a*Ke(19,22)/c
     1- Ke(12,19) + Ke(19,21)*(-a**2 - c**2)/c**2 
     1+ Ke(19,24)*(-a**2 - c**2)/c**2
      Ke(9,20) = 2.0*E*a*v/(3.0*v**2 - 3.0) - a**2*Ke(15,20)/c**2 
     1- a**2*Ke(18,20)/c**2 - 2.0*a*Ke(19,20)/c - 2.0*a*Ke(20,22)/c 
     1- Ke(12,20) + Ke(20,21)*(-a**2 - c**2)/c**2 
     1+ Ke(20,24)*(-a**2 - c**2)/c**2
      Ke(9,21) = -a**2*Ke(15,21)/c**2 - a**2*Ke(18,21)/c**2 
     1- 2.0*a*Ke(19,21)/c - 2.0*a*Ke(21,22)/c - Ke(12,21) 
     1+ Ke(21,21)*(-a**2 - c**2)/c**2 + Ke(21,24)*(-a**2 - c**2)/c**2
      Ke(9,22) = -2.0*E*b/(3.0*v**2 - 3.0) - a**2*Ke(15,22)/c**2 
     1- a**2*Ke(18,22)/c**2 - 2.0*a*Ke(19,22)/c - 2.0*a*Ke(22,22)/c 
     1- Ke(12,22) + Ke(21,22)*(-a**2 - c**2)/c**2 
     1+ Ke(22,24)*(-a**2 - c**2)/c**2
      Ke(9,23) = -2.0*E*a*v/(3.0*v**2 - 3.0) - a**2*Ke(15,23)/c**2 
     1- a**2*Ke(18,23)/c**2 - 2.0*a*Ke(19,23)/c - 2.0*a*Ke(22,23)/c 
     1- Ke(12,23) + Ke(21,23)*(-a**2 - c**2)/c**2 
     1+ Ke(23,24)*(-a**2 - c**2)/c**2
      Ke(9,24) = -a**2*Ke(15,24)/c**2 - a**2*Ke(18,24)/c**2 
     1- 2.0*a*Ke(19,24)/c - 2.0*a*Ke(22,24)/c - Ke(12,24) 
     1+ Ke(21,24)*(-a**2 - c**2)/c**2 + Ke(24,24)*(-a**2 - c**2)/c**2
      Ke(10,10) = -2.0*E*a*c/(3.0*b*v + 3.0*b) + a**2*Ke(15,15)/c**2 
     1+ 2.0*a**2*Ke(15,24)/c**2 + a**2*Ke(24,24)/c**2 
     1+ 2.0*a**2*Ke(15,20)/(b*c) + 2.0*a**2*Ke(15,23)/(b*c) 
     1+ 2.0*a**2*Ke(20,24)/(b*c) + 2.0*a**2*Ke(23,24)/(b*c) 
     1+ a**2*Ke(20,20)/b**2 + 2.0*a**2*Ke(20,23)/b**2 
     1+ a**2*Ke(23,23)/b**2 + 2.0*a*Ke(15,22)/c + 2.0*a*Ke(22,24)/c 
     1+ 2.0*a*Ke(20,22)/b + 2.0*a*Ke(22,23)/b + Ke(22,22)
      Ke(10,11) = -E*c/(3.0*v + 3.0) - a*Ke(11,15)/c - a*Ke(11,24)/c 
     1- a*Ke(11,20)/b - a*Ke(11,23)/b - Ke(11,22)
      Ke(10,12) = -a*Ke(12,15)/c - a*Ke(12,24)/c - a*Ke(12,20)/b 
     1- a*Ke(12,23)/b - Ke(12,22)
      Ke(10,13) = -a**2*Ke(15,15)/c**2 - a**2*Ke(15,18)/c**2 
     1- a**2*Ke(15,21)/c**2 - 2.0*a**2*Ke(15,24)/c**2 
     1- a**2*Ke(18,24)/c**2 - a**2*Ke(21,24)/c**2 - a**2*Ke(24,24)/c**2 
     1- a**2*Ke(15,20)/(b*c) -a**2*Ke(15,23)/(b*c)-a**2*Ke(18,20)/(b*c) 
     1- a**2*Ke(18,23)/(b*c) - a**2*Ke(20,21)/(b*c)-a**2*Ke(20,24)/(b*c)
     1- a**2*Ke(21,23)/(b*c) - a**2*Ke(23,24)/(b*c) + a*Ke(15,16)/c 
     1- a*Ke(15,19)/c - 2.0*a*Ke(15,22)/c + a*Ke(16,24)/c-a*Ke(18,22)/c 
     1- a*Ke(19,24)/c - a*Ke(21,22)/c - 2.0*a*Ke(22,24)/c+a*Ke(16,20)/b 
     1 + a*Ke(16,23)/b - a*Ke(19,20)/b - a*Ke(19,23)/b - a*Ke(20,22)/b 
     1- a*Ke(22,23)/b + Ke(16,22) - Ke(19,22) - Ke(22,22) + (E*a**2*c*v
     1- E*a**2*c - E*b**2*c)/(6.0*a*b*v**2 - 6.0*a*b)
      Ke(10,14) = -7.0*E*c/(12.0*v + 12.0) + a*Ke(15,17)/c+a*Ke(15,20)/c
     1+ a*Ke(15,23)/c + a*Ke(17,24)/c + a*Ke(20,24)/c + a*Ke(23,24)/c 
     1+ a*Ke(17,20)/b + a*Ke(17,23)/b + a*Ke(20,20)/b +2.0*a*Ke(20,23)/b
     1+ a*Ke(23,23)/b + Ke(17,22) + Ke(20,22) + Ke(22,23)
      Ke(10,15) = -a*Ke(15,15)/c - a*Ke(15,24)/c - a*Ke(15,20)/b 
     1- a*Ke(15,23)/b - Ke(15,22)
      Ke(10,16) = -E*a*c/(6.0*b*v + 6.0*b) - a*Ke(15,16)/c-a*Ke(16,24)/c
     1- a*Ke(16,20)/b - a*Ke(16,23)/b - Ke(16,22)
      Ke(10,17) = E*c/(6.0*v + 6.0) - a*Ke(15,17)/c - a*Ke(17,24)/c 
     1- a*Ke(17,20)/b - a*Ke(17,23)/b - Ke(17,22)
      Ke(10,18) = -a*Ke(15,18)/c - a*Ke(18,24)/c - a*Ke(18,20)/b
     1- a*Ke(18,23)/b - Ke(18,22)
      Ke(10,19) = -E*a*c/(3.0*b*v + 3.0*b) - a*Ke(15,19)/c-a*Ke(19,24)/c
     1- a*Ke(19,20)/b - a*Ke(19,23)/b - Ke(19,22)
      Ke(10,20) = E*c/(3.0*v + 3.0) - a*Ke(15,20)/c - a*Ke(20,24)/c 
     1- a*Ke(20,20)/b - a*Ke(20,23)/b - Ke(20,22)
      Ke(10,21) = -a*Ke(15,21)/c - a*Ke(21,24)/c - a*Ke(20,21)/b 
     1- a*Ke(21,23)/b - Ke(21,22)
      Ke(10,22) = E*a*c/(3.0*b*v + 3.0*b) - a*Ke(15,22)/c 
     1- a*Ke(22,24)/c - a*Ke(20,22)/b - a*Ke(22,23)/b - Ke(22,22)
      Ke(10,23) = E*c/(3.0*v + 3.0) - a*Ke(15,23)/c - a*Ke(23,24)/c 
     1- a*Ke(20,23)/b - a*Ke(23,23)/b - Ke(22,23)
      Ke(10,24) = -a*Ke(15,24)/c - a*Ke(24,24)/c - a*Ke(20,24)/b 
     1- a*Ke(23,24)/b - Ke(22,24)
      Ke(11,13) = E*c*v/(6.0*v**2 - 6.0) + a*Ke(11,15)/c + a*Ke(11,18)/c
     1+ a*Ke(11,21)/c + a*Ke(11,24)/c - Ke(11,16) + Ke(11,19) +Ke(11,22)
      Ke(11,14) = -E*b*c/(4.0*a*v + 4.0*a) - Ke(11,17) 
     1- Ke(11,20) - Ke(11,23)
      Ke(12,13) = a*Ke(12,15)/c + a*Ke(12,18)/c + a*Ke(12,21)/c 
     1+ a*Ke(12,24)/c - Ke(12,16) + Ke(12,19) + Ke(12,22)
      Ke(12,14) = -Ke(12,17) - Ke(12,20) - Ke(12,23)
      Ke(13,13) = E*b*c/(3.0*a*v**2 - 3.0*a) + a**2*Ke(15,15)/c**2 
     1+ 2.0*a**2*Ke(15,18)/c**2 + 2.0*a**2*Ke(15,21)/c**2 
     1+ 2.0*a**2*Ke(15,24)/c**2 + a**2*Ke(18,18)/c**2 
     1+ 2.0*a**2*Ke(18,21)/c**2 + 2.0*a**2*Ke(18,24)/c**2 
     1+ a**2*Ke(21,21)/c**2 + 2.0*a**2*Ke(21,24)/c**2 
     1+ a**2*Ke(24,24)/c**2 - 2.0*a*Ke(15,16)/c + 2.0*a*Ke(15,19)/c 
     1+ 2.0*a*Ke(15,22)/c - 2.0*a*Ke(16,18)/c - 2.0*a*Ke(16,21)/c 
     1- 2.0*a*Ke(16,24)/c + 2.0*a*Ke(18,19)/c + 2.0*a*Ke(18,22)/c 
     1+ 2.0*a*Ke(19,21)/c + 2.0*a*Ke(19,24)/c + 2.0*a*Ke(21,22)/c 
     1+ 2.0*a*Ke(22,24)/c + Ke(16,16) - 2.0*Ke(16,19) - 2.0*Ke(16,22) 
     1+ Ke(19,19) + 2.0*Ke(19,22) + Ke(22,22)
      Ke(13,14) = -a*Ke(15,17)/c - a*Ke(15,20)/c - a*Ke(15,23)/c 
     1- a*Ke(17,18)/c - a*Ke(17,21)/c - a*Ke(17,24)/c - a*Ke(18,20)/c 
     1- a*Ke(18,23)/c - a*Ke(20,21)/c - a*Ke(20,24)/c - a*Ke(21,23)/c 
     1- a*Ke(23,24)/c + Ke(16,17) + Ke(16,20) + Ke(16,23) - Ke(17,19) 
     1- Ke(17,22) - Ke(19,20) - Ke(19,23) - Ke(20,22) - Ke(22,23) 
     1+ (E*c*v - 3.0*E*c)/(12.0*v**2 - 12.0)
      Ke(13,15) = a*Ke(15,15)/c + a*Ke(15,18)/c + a*Ke(15,21)/c 
     1+ a*Ke(15,24)/c - Ke(15,16) + Ke(15,19) + Ke(15,22)
      Ke(13,16) = -E*b*c/(6.0*a*v**2 - 6.0*a) + a*Ke(15,16)/c 
     1+ a*Ke(16,18)/c + a*Ke(16,21)/c + a*Ke(16,24)/c - Ke(16,16) 
     1+ Ke(16,19) + Ke(16,22)
      Ke(13,17) = E*c*v/(6.0*v**2 - 6.0) + a*Ke(15,17)/c + a*Ke(17,18)/c
     1+ a*Ke(17,21)/c + a*Ke(17,24)/c - Ke(16,17) + Ke(17,19) +Ke(17,22)
      Ke(13,18) = a*Ke(15,18)/c + a*Ke(18,18)/c + a*Ke(18,21)/c 
     1+ a*Ke(18,24)/c - Ke(16,18) + Ke(18,19) + Ke(18,22)
      Ke(13,19) = E*b*c/(6.0*a*v**2 - 6.0*a) + a*Ke(15,19)/c 
     1+ a*Ke(18,19)/c + a*Ke(19,21)/c + a*Ke(19,24)/c - Ke(16,19) 
     1+ Ke(19,19) + Ke(19,22)
      Ke(13,20) = -E*c*v/(6.0*v**2 - 6.0) + a*Ke(15,20)/c 
     1+ a*Ke(18,20)/c + a*Ke(20,21)/c + a*Ke(20,24)/c - Ke(16,20) 
     1+ Ke(19,20) + Ke(20,22)
      Ke(13,21) = a*Ke(15,21)/c + a*Ke(18,21)/c + a*Ke(21,21)/c 
     1+ a*Ke(21,24)/c - Ke(16,21) + Ke(19,21) + Ke(21,22)
      Ke(13,22) = E*b*c/(6.0*a*v**2 - 6.0*a) + a*Ke(15,22)/c 
     1+ a*Ke(18,22)/c + a*Ke(21,22)/c + a*Ke(22,24)/c - Ke(16,22) 
     1+ Ke(19,22) + Ke(22,22)
      Ke(13,23) = E*c*v/(6.0*v**2 - 6.0) + a*Ke(15,23)/c +a*Ke(18,23)/c 
     1+ a*Ke(21,23)/c + a*Ke(23,24)/c - Ke(16,23) +Ke(19,23) + Ke(22,23)
      Ke(13,24) = a*Ke(15,24)/c + a*Ke(18,24)/c + a*Ke(21,24)/c 
     1+ a*Ke(24,24)/c - Ke(16,24) + Ke(19,24) + Ke(22,24)
      Ke(14,14) = -E*b*c/(2.0*a*v + 2.0*a) + Ke(17,17) + 2.0*Ke(17,20) 
     1+ 2.0*Ke(17,23) + Ke(20,20) + 2.0*Ke(20,23) + Ke(23,23)
      Ke(14,15) = -Ke(15,17) - Ke(15,20) - Ke(15,23)
      Ke(14,16) = -E*c/(4.0*v + 4.0) - Ke(16,17) - Ke(16,20) - Ke(16,23)
      Ke(14,17) = E*b*c/(4.0*a*v + 4.0*a) - Ke(17,17) 
     1- Ke(17,20) - Ke(17,23)
      Ke(14,18) = -Ke(17,18) - Ke(18,20) - Ke(18,23)
      Ke(14,19) = -E*c/(4.0*v + 4.0) - Ke(17,19) - Ke(19,20) - Ke(19,23)
      Ke(14,20) = E*b*c/(4.0*a*v + 4.0*a) - Ke(17,20) 
     1- Ke(20,20) - Ke(20,23)
      Ke(14,21) = -Ke(17,21) - Ke(20,21) - Ke(21,23)
      Ke(14,22) = E*c/(4.0*v + 4.0) - Ke(17,22) - Ke(20,22) - Ke(22,23)
      Ke(14,23) = E*b*c/(4.0*a*v + 4.0*a) - Ke(17,23) 
     1- Ke(20,23) - Ke(23,23)
      Ke(14,24) = -Ke(17,24) - Ke(20,24) - Ke(23,24)
      !print *, 'a=',a
      !print *, 'b=',b
      !print *, 'E=',E
      !print *, 'v=',v
      
      
      
      do i=1,24
              do j=1,24
                  if(i .gt. j) then 
                      Ke(i,j)=Ke(j,i)
                  elseif(i .le. j) then
                      exit
                  endif 
              enddo
      enddo
!c----------------------------------------------------
!      call SYSTEM_CLOCK(time1)
      !write(*,*) 'time1--',time1
      !write(*,*) 'inputs--',inputs
      !write(*,*) 'targets--',targets
      
      !write(*,"(5(F16.4))") miu,length,width,ratio
      end subroutine ANN
      
      Subroutine revolve(nodes,Te,TransposeTe)
      Implicit none
      integer, parameter :: n = 3
      real, dimension(n,n) :: a, q, r
      integer flag,i,j,k
      real*8 nodes(3,8),u(3,1)
      real*8 Te(24,24),TransposeTe(24,24)
      real*8 F(3,3),F1(3),F2(3),F3(3)
      real*8 detx1,detx2,detx3,dety1,dety2,dety3,detz1,detz2,detz3
      
      detx1=0.125*((nodes(1,5)-nodes(1,1))*(1+1)*(1+1)
     1+(nodes(1,6)-nodes(1,2))*(1-1)*(1+1)
     1+(nodes(1,7)-nodes(1,3))*(1-1)*(1-1)
     1+(nodes(1,8)-nodes(1,4))*(1+1)*(1-1))
      detx2=0.125*((nodes(1,1)-nodes(1,2))*(1+1)*(1+1)
     1+(nodes(1,4)-nodes(1,3))*(1+1)*(1-1)
     1+(nodes(1,5)-nodes(1,6))*(1-1)*(1+1)
     1+(nodes(1,8)-nodes(1,7))*(1-1)*(1-1))
      detx3=0.125*((nodes(1,1)-nodes(1,4))*(1+1)*(1+1)
     1+(nodes(1,2)-nodes(1,3))*(1+1)*(1-1)
     1+(nodes(1,5)-nodes(1,8))*(1-1)*(1+1)
     1+(nodes(1,6)-nodes(1,7))*(1-1)*(1-1))
      dety1=0.125*((nodes(2,5)-nodes(2,1))*(1+1)*(1+1)
     1+(nodes(2,6)-nodes(2,2))*(1-1)*(1+1)
     1+(nodes(2,7)-nodes(2,3))*(1-1)*(1-1)
     1+(nodes(2,8)-nodes(2,4))*(1+1)*(1-1))
      dety2=0.125*((nodes(2,1)-nodes(2,2))*(1+1)*(1+1)
     1+(nodes(2,4)-nodes(2,3))*(1+1)*(1-1)
     1+(nodes(2,5)-nodes(2,6))*(1-1)*(1+1)
     1+(nodes(2,8)-nodes(2,7))*(1-1)*(1-1))
      dety3=0.125*((nodes(2,1)-nodes(2,4))*(1+1)*(1+1)
     1+(nodes(2,2)-nodes(2,3))*(1+1)*(1-1)
     1+(nodes(2,5)-nodes(2,8))*(1-1)*(1+1)
     1+(nodes(2,6)-nodes(2,7))*(1-1)*(1-1))
      detz1=0.125*((nodes(3,5)-nodes(3,1))*(1+1)*(1+1)
     1+(nodes(3,6)-nodes(3,2))*(1-1)*(1+1)
     1+(nodes(3,7)-nodes(3,3))*(1-1)*(1-1)
     1+(nodes(3,8)-nodes(3,4))*(1+1)*(1-1))
      detz2=0.125*((nodes(3,1)-nodes(3,2))*(1+1)*(1+1)
     1+(nodes(3,4)-nodes(3,3))*(1+1)*(1-1)
     1+(nodes(3,5)-nodes(3,6))*(1-1)*(1+1)
     1+(nodes(3,8)-nodes(3,7))*(1-1)*(1-1))
      detz3=0.125*((nodes(3,1)-nodes(3,4))*(1+1)*(1+1)
     1+(nodes(3,2)-nodes(3,3))*(1+1)*(1-1)
     1+(nodes(3,5)-nodes(3,8))*(1-1)*(1+1)
     1+(nodes(3,6)-nodes(3,7))*(1-1)*(1-1))
      F(1,1)=detx1
      F(2,1)=dety1
      F(3,1)=detz1
      F(1,2)=detx2
      F(2,2)=dety2
      F(3,2)=detz2
      F(1,3)=detx3
      F(2,3)=dety3
      F(3,3)=detz3
      !print *, 'F=',F
      !F(1,1)=0
      !F(2,1)=0
      !F(3,1)=2
      !F(1,2)=-1
      !F(2,2)=0
      !F(3,2)=0
      !F(1,3)=0
      !F(2,3)=1
      !F(3,3)=0
      
      F1(:)=F(:,1)
      !print *, 'F1=',F1
      F2(:)=F(:,2)-((F(1,1)*F(1,2)+F(2,1)*F(2,2)+F(3,1)*F(3,2))
     1/(F(1,1)*F(1,1)+F(2,1)*F(2,1)+F(3,1)*F(3,1)))*F(:,1)
      !print *, 'F2=',F2
      F3(:)=F(:,3)-((F(1,1)*F(1,3)+F(2,1)*F(2,3)+F(3,1)*F(3,3))
     1/(F(1,1)*F(1,1)+F(2,1)*F(2,1)+F(3,1)*F(3,1)))*F(:,1)
     1-((F(1,2)*F(1,3)+F(2,2)*F(2,3)+F(3,2)*F(3,3))
     1/(F(1,2)*F(1,2)+F(2,2)*F(2,2)+F(3,2)*F(3,2)))*F(:,2)
      !print *, 'F3=',F3
      F1(:)=F1(:)/SQRT(F1(1)**2+F1(2)**2+F1(3)**2)
      F2(:)=F2(:)/SQRT(F2(1)**2+F2(2)**2+F2(3)**2)
      F3(:)=F3(:)/SQRT(F3(1)**2+F3(2)**2+F3(3)**2)
      Q(:,1)=F1(:)
      Q(:,2)=F2(:)
      Q(:,3)=F3(:)
      
      Te(:,:)=0
      TransposeTe(:,:)=0
      
      do k=1,8
        Te(3*k-2,1+(k-1)*3) = Q(1,1)
        Te(3*k-1,1+(k-1)*3) = Q(2,1)
        Te(3*k,  1+(k-1)*3) = Q(3,1)
        Te(3*k-2,2+(k-1)*3) = Q(1,2)
        Te(3*k-1,2+(k-1)*3) = Q(2,2)
        Te(3*k,  2+(k-1)*3) = Q(3,2)
        Te(3*k-2,3+(k-1)*3) = Q(1,3)
        Te(3*k-1,3+(k-1)*3) = Q(2,3)
        Te(3*k,  3+(k-1)*3) = Q(3,3)  
      enddo
      !print *, 'Te=',Te
      do i = 1, 24
        do j = 1, 24
            TransposeTe(j, i) = Te(i, j)
        end do
      end do
      
      END subroutine revolve
      
      
      SUBROUTINE UEL(RHS,AMATRX,SVARS,ENERGY,NDOFEL,NRHS,NSVARS,
     1     PROPS,NPROPS,COORDS,MCRD,NNODE,U,DU,V,A,JTYPE,TIME,
     2     DTIME,KSTEP,KINC,JELEM,PARAMS,NDLOAD,JDLTYP,ADLMAG,
     3     PREDEF,NPREDF,LFLAGS,MLVARX,DDLMAG,MDLOAD,PNEWDT,
     4     JPROPS,NJPROP,PERIOD)
C     
      INCLUDE 'ABA_PARAM.INC'
      PARAMETER ( ZERO = 0.D0, HALF = 0.5D0, ONE = 1.D0 )
C
      DIMENSION RHS(MLVARX,*),AMATRX(NDOFEL,NDOFEL),
     1     SVARS(NSVARS),ENERGY(8),PROPS(*),COORDS(MCRD,NNODE),
     2     U(NDOFEL),DU(MLVARX,*),V(NDOFEL),A(NDOFEL),TIME(2),
     3     PARAMS(3),JDLTYP(MDLOAD,*),ADLMAG(MDLOAD,*),
     4     DDLMAG(MDLOAD,*),PREDEF(2,NPREDF,NNODE),LFLAGS(*),
     5     JPROPS(*)
      DIMENSION SRESID(8)
      real*8 E,length,width,ratio,inputs(2),nodes(3,8)
      real*8 COORDS_sort(MCRD,NNODE),xyDU(24,1),Ke(24,24)
      real*8 miu,t,Te(24,24),TransposeTe(24,24),Ke2(24,24)
      real*8 upa(3),upb(3),upc(3),upd(3)
      real*8 downa(3),downb(3),downc(3),downd(3)
      
      nodes(:,:)=COORDS(:,:)
	!Obtain the length and width and ratio
      
      upa(:)=(COORDS(:,1)+COORDS(:,2))*0.5
      upb(:)=(COORDS(:,2)+COORDS(:,6))*0.5
      upc(:)=(COORDS(:,6)+COORDS(:,5))*0.5
      upd(:)=(COORDS(:,5)+COORDS(:,1))*0.5
      
      downa(:)=(COORDS(:,4)+COORDS(:,3))*0.5
      downb(:)=(COORDS(:,3)+COORDS(:,7))*0.5
      downc(:)=(COORDS(:,7)+COORDS(:,8))*0.5
      downd(:)=(COORDS(:,8)+COORDS(:,4))*0.5
      
      length=0.5*(SQRT((upa(1) - upc(1)) ** 2 + (upa(2) - upc(2)) ** 2
     1+ (upa(3) - upc(3)) ** 2)+SQRT((downa(1) - downc(1)) ** 2 
     1+ (downa(2) - downc(2)) ** 2+ (downa(3) - downc(3)) ** 2))
      
      width=0.5*(SQRT((upb(1) - upd(1)) ** 2 + (upb(2) - upd(2)) ** 2
     1+ (upb(3) - upd(3)) ** 2)+SQRT((downb(1) - downd(1)) ** 2 
     1+ (downb(2) - downd(2)) ** 2+ (downb(3) - downd(3)) ** 2))
      
      
      !aspect ratio
      ratio=length/width
      miu = PROPS(3)
      E = PROPS(2)
      t = 0.5*PROPS(1)
      
      inputs(1)=ratio
      inputs(2)=miu
      
      call ANN(t,E,miu,length,width,inputs,PROPS,Ke)
      call revolve(nodes,Te,TransposeTe)
      
      
      DO K1 = 1, NDOFEL
      DO KRHS = 1, NRHS
          RHS(K1,KRHS) = ZERO

      END DO
      END DO
      Ke=matmul(Te,Ke)
      
      Ke=matmul(Ke,TransposeTe)
      
      IF (LFLAGS(3).EQ.1) THEN
C       Normal incrementation
        IF (LFLAGS(1).EQ.1 .OR. LFLAGS(1).EQ.2) THEN
C         *STATIC
          AMATRX =  Ke
          IF (LFLAGS(4).NE.0) THEN   
          ELSE
            xyDU(1:24,1)=DU(1:24,1) 
          DO I=1,NDOFEL
            RHS(I,1) = RHS(I,1)-sum(matmul(Ke(I,1:24),xyDU))
          ENDDO 
              !print *, 'ERROR1='
          ENDIF 
        ELSE
            !print *, 'ERROR2='
        ENDIF   
      ENDIF

           
            
      
      RETURN
      END

      
      