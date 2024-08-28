#include "imax.cuh"

#define BLOQUE 128

__global__
static void d_kerd__imax(
	uint x0_t, uint X0, float * x0, float * dx0,
	//
	uint    Y,
	float * y, float * dy,
	//
	uint mega_t,
	//
	uint _c0,
	//
	uint C0)
{
	uint _y = threadIdx.x + blockIdx.x * blockDim.x;
	uint _t = threadIdx.y + blockIdx.y * blockDim.y;
	//
	//if (_y < Y && _t < GRAND_T) {
	uint tx0 = t_MODE(_t, mega_t-x0_t);
	uint ty  = t_MODE(_t, mega_t     );
	//
	//__shared__ float __x_[128];
	__shared__ float _dy_[1];
	__shared__ float __y_[1];
	//if ( _y < (X0/C0) ) _x_[threadIdx.x] = x0[tx0*X0 + _y];
	//else                _x_[threadIdx.x] = -1e35.f;
	//__syncthreads();
	//
	/*uint lg = (uint)log2f((float)Vect);
	FOR(1, l, lg+1) {
		//uint p = pow(2, l);
		uint p = 1 << l;
		uint p1 = 1 << (l-1);
		if (thx % p == 0) _x_[thx] = MAX2(_x_[thx], _x_[thx+p1]);
		__syncthreads();
	}
	__syncthreads();*/
	//
	if (threadIdx.x == 0 && _y<Y) {
		//atomicMax(&y[ty*Y + _c0], _x_[0]);
		__y_[0] =  y[ty*Y + _c0];
		_dy_[0] = dy[ty*Y + _c0];
	}
	__syncthreads();

	if (x0[tx0*X0 + _y] == __y_[0]) {
		atomicAdd(&dx0[tx0*X0 + _y], _dy_[0]);
	}
};

void imax__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t) {
	uint X0 = inst->x_Y[0]; uint x0_t = inst->x_t[0];
	//
	uint \
		C0 = inst->params[0];
	//
	bool x0_existe = (mega_t != 0 ? true : (x0_t != 1));
	//
	//printf("%i %i\n", X0,C0);
	ASSERT(BLOQUE <= (X0/C0));
	//
	if (x0_existe) {
		FOR(0, _c0, C0) {
			d_kerd__imax<<<dim3(KERD((X0/C0),BLOQUE),  KERD(GRAND_T,1)), dim3(BLOQUE,1)>>>(
				inst->x_t[0], inst->x_Y[0], x__d[0], dx__d[0],
				//
				inst->Y,
				inst->y__d, inst->dy__d,
				//
				mega_t,
				//
				_c0,
				//
				C0
			);
		}
	} else {
		//	rien
	}
};