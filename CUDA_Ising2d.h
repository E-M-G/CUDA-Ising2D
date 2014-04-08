#include <cstdlib>
#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace std;


class Ising2d {

	public:
		
		Ising2d(int m, int matrixType);
		void calcStatesEnergy();
		double getStatesEnergy();
		double getChangeEnergy(int i, int j);
		double getStatesMag();
		int spin(int i , int j);
		void spin(int i , int j , int value);
		void printMatrix();
		inline void flipSpin(int i , int j);

		//CUDA STUFF

		extern void CUDA_Constructor(int** d_state_array, int size);
		extern void CUDA_setArray(int* d_state_array, int* h_state_array, int size);
		extern void CUDA_getArray(int *h_state_array, int* d_state_array, int size);
		extern int CUDA_getStatesEnergy(void);

		int *d_state_array, *h_state_array;
		int m_size;

		void initState(int size);
		void setState(int *h_state_array){ CUDA_setArray(d_state_array, h_state_array , size);}
		

		
	private:
		
		vector<int> m_state;
		vector<int> m_ene;

		int m_m;
		double m_stateEne;
		double m_changeEne;
		double m_mag;
		static const double m_j = 1.0; //interaction strength

		void createNewMatrix(int spinValue);


};

inline void Ising2d::flipSpin(int i , int j){

	m_state[i*m_m +j] = m_state[i*m_m + j ]*-1;
}

void initState(int size){

	M_size = size;
	CUDA_Constructor(&d_state_array, size);
	h_state_array = (int*)malloc(sizeof(int)*size);
}