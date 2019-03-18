MPICXX = mpicxx
CXX = g++ 
OPT = -O2
LIBS = -lboost_program_options -lfftw3 -lopenblas -llapack -lpthread -lgfortran 

all : exact_2d fssh_nd_mpi fssh_nd_rescalex_mpi

exact_2d: exact_2d.cpp 
	$(CXX) $(OPT) $< -o $@ $(LIBS)

fssh_nd_mpi: fssh_nd_mpi.cpp 
	$(MPICXX) $(OPT) $< -o $@ $(LIBS)

fssh_nd_rescalex_mpi: fssh_nd_rescalex_mpi.cpp 
	$(MPICXX) $(OPT) $< -o $@ $(LIBS)
