#include "sous_scal.cuh"

#define BLOQUE 128

static __global__ void d_kerd__sous_scal__simple(
	uint x0_t, uint X0, float * x0, float * dx0,
	uint x1_t, uint X1, float * x1, float * dx1,
	//
	uint    Y,
	float * y, float * dy,
	//
	uint * ts__d, uint mega_t,
	//
	//uint _c0,
	//
	uint C0)
{
	uint _tc0 = threadIdx.x + blockIdx.x * blockDim.x;
	uint _y   = threadIdx.y + blockIdx.y * blockDim.y;

	uint _c0 = (_tc0 - (_tc0%GRAND_T))/GRAND_T;
	uint _t  = _tc0 - _c0*GRAND_T;

	//if (_ay < Ay && _c0 < C0 && _t < GRAND_T) {
	uint tx0 = t_MODE(_t, mega_t-x0_t);
	uint tx1 = t_MODE(_t, mega_t-x1_t);
	uint ty  = t_MODE(_t, mega_t     );
	//
	uint Vect = X0/C0;
	//
	__shared__ float _d_sous_;
	if (threadIdx.x == 0 && _y<Vect) _d_sous_ = 0;
	__syncthreads();

	if (_y < Vect) {
		float _dy = dy[ty*Y + _c0*Vect + _y];
		//y[ty*Y + _c0*Vect + _y] = x0[tx0*X0 + _c0*Vect + _y] - _sous_;
		atomicAdd(&dx0[tx0*X0 + _c0*Vect + _y], _dy);
		atomicAdd(&_d_sous_, -_dy);
	}
	__syncthreads();

	if (threadIdx.x == 0 && _y<Vect) atomicAdd(&dx1[tx1*X1 + _c0], _d_sous_);
};

//	---------------------------------------------------------------------------------

void sous_scal__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t) {
	uint * params = inst->params;
	uint \
		C0 =params[0];
	//
	uint x0_t = inst->x_t[0];
	uint x1_t = inst->x_t[1];
	//
	bool x0_existe = (mega_t != 0 ? true : (x0_t != 1));
	bool x1_existe = (mega_t != 0 ? true : (x1_t != 1));
	//
	ASSERT(x0_existe && x1_existe);
	//
	uint Vect = inst->Y / C0;
	//
	ASSERT(BLOQUE <= Vect);
	//
	if (x0_existe && x1_existe) {
		//FOR(0, _c0, C0) {
			d_kerd__sous_scal__simple<<<dim3(KERD((GRAND_T*C0),1), KERD(Vect,BLOQUE)), dim3(1,BLOQUE,1)>>>(
				inst->x_t[0], inst->x_Y[0], x__d[0], dx__d[0],
				inst->x_t[1], inst->x_Y[1], x__d[1], dx__d[1],
				//
				inst->Y,
				inst->y__d, inst->dy__d,
				//
				ts__d, mega_t,
				//
		//		_c0,
				//
				C0
			);
		//}
	} else {
		inst_zero_mega_t(inst, mega_t);
	}
};