program test_pce_forward
  implicit none

  real*8 :: inputs(2), outputs(78)
  integer :: i, j
  character(len=100) :: filename
  real*8 :: test_inputs(10,2)
  real*8 :: start_time, end_time

  ! 测试多组输入数据
  test_inputs(1,:) = (/ 0.3d0, -0.5d0 /)
  test_inputs(2,:) = (/ -0.2d0, 0.8d0 /)
  test_inputs(3,:) = (/ 0.0d0, 0.0d0 /)
  test_inputs(4,:) = (/ 1.0d0, 1.0d0 /)
  test_inputs(5,:) = (/ -1.0d0, -1.0d0 /)
  test_inputs(6,:) = (/ 0.5d0, 0.5d0 /)
  test_inputs(7,:) = (/ -0.8d0, 0.3d0 /)
  test_inputs(8,:) = (/ 0.7d0, -0.9d0 /)
  test_inputs(9,:) = (/ 0.1d0, 0.9d0 /)
  test_inputs(10,:) = (/ -0.4d0, -0.6d0 /)

  print *, '========================================='
  print *, 'PCE Neural Network Replacement Demo'
  print *, 'Input dimension: 2, Output dimension: 78'
  print *, '========================================='

  ! 测试推理速度
  call cpu_time(start_time)
  do j = 1, 1000
     do i = 1, 10
        call pce_forward(test_inputs(i,:), outputs)
     enddo
  enddo
  call cpu_time(end_time)

  print *, 'Performance test (10000 inferences):'
  print *, 'Total time: ', end_time - start_time, ' seconds'
  print *, 'Average time per inference: ', (end_time - start_time) / 10000.0d0 * 1000.0d0, ' ms'
  print *, ''

  ! 显示几个测试样例的结果
  do i = 1, 5
     call pce_forward(test_inputs(i,:), outputs)
     print *, 'Test case ', i, ':'
     print *, 'Input: [', test_inputs(i,1), ', ', test_inputs(i,2), ']'
     print *, 'First 5 outputs: ', outputs(1:5)
     print *, 'Last 5 outputs: ', outputs(74:78)
     print *, ''
  enddo

  ! 保存结果到文件
  open(unit=10, file='pce_results.txt', status='replace')
  write(10, *) 'PCE Neural Network Replacement Results'
  write(10, *) 'Input_1, Input_2, Output_1, Output_2, ..., Output_78'
  do i = 1, 10
     call pce_forward(test_inputs(i,:), outputs)
     write(10, '(80F12.6)') test_inputs(i,:), outputs
  enddo
  close(10)

  print *, 'Results saved to pce_results.txt'

end program test_pce_forward


subroutine pce_forward(inputs, outputs)
  implicit none

  real*8, intent(in) :: inputs(2)
  real*8, intent(out) :: outputs(78)

  real*8 x1, x2, phi(6)
  integer i
  logical, save :: coeffs_loaded = .false.
  real*8, save :: coeff(78,6)

  ! 第一次调用时从文件读取系数
  if (.not. coeffs_loaded) then
     call load_pce_coefficients(coeff)
     coeffs_loaded = .true.
  endif

  x1 = inputs(1)
  x2 = inputs(2)

  phi(1) = 1.0d0
  phi(2) = x1
  phi(3) = x2
  phi(4) = x1**2
  phi(5) = x1 * x2
  phi(6) = x2**2

  do i = 1, 78
     outputs(i) = sum(coeff(i,1:6) * phi(1:6))
  enddo

end subroutine pce_forward

subroutine load_pce_coefficients(coeff)
  implicit none

  real*8, intent(out) :: coeff(78,6)
  integer :: i, j, ios
  character(len=100) :: filename
  logical :: file_exists

  ! 尝试从文件读取系数
  filename = 'final_pce_coefficients.txt'
  inquire(file=filename, exist=file_exists)

  if (file_exists) then
     print *, 'Loading PCE coefficients from file: ', trim(filename)
     open(unit=20, file=filename, status='old', iostat=ios)

     if (ios == 0) then
        ! 跳过注释行
        do i = 1, 10
           read(20, *, iostat=ios)
           if (ios /= 0) exit
        enddo

        ! 读取系数矩阵
        do i = 1, 78
           read(20, *, iostat=ios) (coeff(i,j), j=1,6)
           if (ios /= 0) then
              print *, 'Error reading coefficients at row ', i
              call use_default_coefficients(coeff)
              close(20)
              return
           endif
        enddo
        close(20)
        print *, 'PCE coefficients loaded successfully from file'
     else
        print *, 'Error opening coefficient file, using defaults'
        call use_default_coefficients(coeff)
     endif
  else
     print *, 'Coefficient file not found, using default coefficients'
     call use_default_coefficients(coeff)
  endif

end subroutine load_pce_coefficients

subroutine use_default_coefficients(coeff)
  implicit none

  real*8, intent(out) :: coeff(78,6)
  integer :: i, j

  ! 使用默认系数（示例系数）
  do i = 1, 78
     coeff(i,1) = 0.02709560d0 + i * 0.001d0    ! 常数项
     coeff(i,2) = 0.90135505d0 * (1.0d0 + i * 0.01d0)  ! x1项
     coeff(i,3) = 0.00066962d0 * (1.0d0 + i * 0.005d0) ! x2项
     coeff(i,4) = -0.01123008d0 * (1.0d0 + i * 0.002d0) ! x1^2项
     coeff(i,5) = -0.00422859d0 + i * 0.01d0    ! x1*x2项
     coeff(i,6) = 0.72299447d0 * (1.0d0 + i * 0.001d0)  ! x2^2项
  enddo

  print *, 'Using default PCE coefficients'

end subroutine use_default_coefficients
