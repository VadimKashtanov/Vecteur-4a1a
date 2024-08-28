#include "imax.cuh"

#define BLOQUE 128

__global__
static void kerd__imax(	//Un BLOQUE couvre tout le _c0
	uint x0_t, uint X0, float * x0,
	//
	uint    Y,
	float * y,
	//
	uint mega_t,
	//
	uint C0)
{
	uint thx = threadIdx.x;
	//
	uint _c0 = blockIdx.x;
	uint _t  = blockIdx.y;
	//
	//if (_y < Y && _t < GRAND_T) {
	uint tx0 = t_MODE(_t, mega_t-x0_t);
	uint ty  = t_MODE(_t, mega_t     );
	//
	__shared__ float _x_[BLOQUE];
	__shared__ float _max;
	//
	if (thx == 0) _max = -FLT_MAX;
	//
	//
	FOR(0, partie, KERD((X0/C0), BLOQUE)) {
		//
		uint _y = _c0*(X0/C0) + partie*BLOQUE + thx;
		//
		if ( _y < X0 ) _x_[threadIdx.x] = x0[tx0*X0 + _y];
		else           _x_[threadIdx.x] = -FLT_MAX;
		__syncthreads();
		//
		uint lg = (uint)log2f((float)(X0/C0));
		FOR(1, l, lg+1) {
			//uint p = pow(2, l);
			uint p = 1 << l;
			uint p1 = 1 << (l-1);
			if (threadIdx.x % p == 0) _x_[threadIdx.x] = MAX2(_x_[threadIdx.x], _x_[threadIdx.x+p1]);
			__syncthreads();
		}
		__syncthreads();
		//
		if (threadIdx.x == 0) _max = MAX2(_max, _x_[0]);
	}
	__syncthreads();
	//
	if (threadIdx.x == 0) {
		y[ty*Y + _c0] = _max;
	}
};

void imax__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t, uint entrainnement) {
	uint X0 = inst->x_Y[0]; uint x0_t = inst->x_t[0];
	//
	uint \
		C0    = inst->params[0];
	//
	bool x0_existe = (mega_t != 0 ? true : (x0_t != 1));
	//
	if (x0_existe) {
		//FOR(0, _c0, C0) {
			kerd__imax<<<dim3(KERD(/*(X0/*/C0/*)*/,/*BLOQUE*/1), KERD(GRAND_T,1)), dim3(BLOQUE,1)>>>(
				inst->x_t[0], inst->x_Y[0], x__d[0],
				//
				inst->Y,
				inst->y__d,
				//
				mega_t,
				//
				//_c0,
				//
				C0
			);
		//}
	} else {
		inst_zero_mega_t(inst, mega_t);
	}
};