#include "div.cuh"

__global__
static void d_kerd__div(
	uint x0_t, uint X0, float * x0, float * dx0,
	uint x1_t, uint X1, float * x1, float * dx1,
	//
	uint    Y,
	float * y, float * dy,
	//
	uint * ts__d, uint mega_t)
{
	uint _y = threadIdx.x + blockIdx.x * blockDim.x;
	uint _t = threadIdx.y + blockIdx.y * blockDim.y;
	//
	if (_y < Y && _t < GRAND_T) {
		uint tx0 = t_MODE(_t, mega_t-x0_t);
		uint tx1 = t_MODE(_t, mega_t-x1_t);
		uint ty  = t_MODE(_t, mega_t     );
		//
		//y[ty*Y + _y] = x0[tx0*X0 + _y] / x1[tx1*X1 + _y];
		float __dy = dy[ty*Y + _y];
		atomicAdd(&dx0[tx0*X0 + _y], __dy / x1[tx1*X1 + _y]);
		atomicAdd(&dx1[tx1*X1 + _y], -__dy * x0[tx0*X0 + _y] / powf(x1[tx1*X1 + _y],2));
	}
};

void div__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t) {
	uint X0 = inst->x_Y[0];	uint x0_t = inst->x_t[0];
	uint X1 = inst->x_Y[1];	uint x1_t = inst->x_t[1];
	uint Y  = inst->Y;
	//
	bool x0_existe = (mega_t != 0 ? true : (x0_t != 1));
	bool x1_existe = (mega_t != 0 ? true : (x1_t != 1));
	//
	uint xs_existants = x0_existe + x1_existe;
	//
	if (xs_existants == 2) {
		d_kerd__div<<<dim3(KERD(Y,16), KERD(GRAND_T,16)), dim3(16,16)>>>(
			x0_t, X0, x__d[0], dx__d[0],
			x1_t, X1, x__d[1], dx__d[1],
			//
			inst->Y,
			inst->y__d, inst->dy__d,
			//
			ts__d, mega_t
		);
	} else {
		//	inst_zero_mega_t(inst, mega_t);
	}
};