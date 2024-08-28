#include "drop_vecteur.cuh"

static __global__ void d_kerd_drop_vecteur(
	uint x0_t, uint X0, float * x0, float * dx0,
	//
	uint    Y,
	float * y, float * dy,
	//
	uint mega_t,
	//
	uint VECT, uint POURCENT,
	//
	float * matrice)
{
	uint _y = threadIdx.x + blockIdx.x * blockDim.x;
	uint _t = threadIdx.y + blockIdx.y * blockDim.y;
	//
	if (_y < Y && _t < GRAND_T) {
		uint tx0 = t_MODE(_t, mega_t-x0_t);
		uint ty  = t_MODE(_t, mega_t     );
		//
		//if (matrice[_y] == 0) {
			//y[ty*Y + _y] = 0.0;
		//} else {
			//y[ty*Y + _y] = x0[tx0*X0 + _y];
			atomicAdd(&dx0[tx0*X0 + _y], dy[ty*Y + _y] * matrice[_y]);
		//}
	};
}

void drop_vecteur__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t) {
	uint \
		VECT     = inst->params[0],	\
		POURCENT = inst->params[1];
	//
	float * matrice = (float*)inst->espace_potentiel;
	//
	uint x0_t = inst->x_t[0];
	uint Y  = inst->Y;
	//
	bool x0_existe = (mega_t != 0 ? true : (x0_t != 1));
	//
	if (x0_existe) {
		d_kerd_drop_vecteur<<<dim3(KERD(Y,16), KERD(GRAND_T,16)), dim3(16,16)>>>(
			inst->x_t[0], inst->x_Y[0], x__d[0], dx__d[0],
			//
			inst->Y,
			inst->y__d, inst->dy__d,
			//
			mega_t,
			//
			VECT, POURCENT,
			//
			matrice
		);
	} else {
		//inst_zero_mega_t(inst, mega_t);
	}
};