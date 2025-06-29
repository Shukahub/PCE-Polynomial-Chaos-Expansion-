# Makefile for PCE Fortran Program
# 支持多种Fortran编译器

# 编译器选择
FC = gfortran
# FC = ifort
# FC = pgfortran

# 编译选项
FFLAGS = -O3 -Wall -fcheck=bounds -g
# Intel Fortran选项: FFLAGS = -O3 -check bounds -g
# PGI Fortran选项: FFLAGS = -O3 -Mbounds -g

# 目标文件
TARGET = pce_demo
SOURCE = PCE.for

# 默认目标
all: $(TARGET)

# 编译规则
$(TARGET): $(SOURCE)
	@echo "Compiling PCE Fortran program..."
	$(FC) $(FFLAGS) -o $(TARGET) $(SOURCE)
	@echo "Compilation successful!"
	@echo "Executable: $(TARGET)"

# 运行程序
run: $(TARGET)
	@echo "Running PCE demo..."
	./$(TARGET)

# 性能测试
benchmark: $(TARGET)
	@echo "Running performance benchmark..."
	time ./$(TARGET)

# 清理
clean:
	@echo "Cleaning up..."
	rm -f $(TARGET) *.o *.mod
	@echo "Clean complete!"

# 安装依赖（仅适用于某些系统）
install-deps:
	@echo "Installing Fortran compiler dependencies..."
	# Ubuntu/Debian
	# sudo apt-get install gfortran
	# CentOS/RHEL
	# sudo yum install gcc-gfortran
	# macOS with Homebrew
	# brew install gcc

# 帮助信息
help:
	@echo "PCE Fortran Makefile"
	@echo "Available targets:"
	@echo "  all        - Compile the program (default)"
	@echo "  run        - Compile and run the program"
	@echo "  benchmark  - Run with timing information"
	@echo "  clean      - Remove compiled files"
	@echo "  help       - Show this help message"
	@echo ""
	@echo "Compiler options:"
	@echo "  FC=gfortran  - Use GNU Fortran (default)"
	@echo "  FC=ifort     - Use Intel Fortran"
	@echo "  FC=pgfortran - Use PGI Fortran"
	@echo ""
	@echo "Example usage:"
	@echo "  make"
	@echo "  make run"
	@echo "  make FC=ifort"
	@echo "  make benchmark"

.PHONY: all run benchmark clean install-deps help
