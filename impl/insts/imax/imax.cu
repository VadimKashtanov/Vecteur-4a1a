#include "imax.cuh"

uint imax__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	return 0;
};

uint imax__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	return 0;
};

void imax__init_poids(Inst_t * inst) {
	ASSERT(inst->Y == inst->params[0]);
	ASSERT(inst->x_Y[0] % inst->params[0] == 0);
	//inst->p__d;
};

static __global__ void _FLT_MIN(
	uint    Y,
	float * y,
	//
	uint mega_t,
	//
	uint C0)
{
	uint _y = threadIdx.x + blockIdx.x * blockDim.x;
	uint _t = threadIdx.y + blockIdx.y * blockDim.y;
	//
	if (_y < Y && _t < GRAND_T) {
		uint ty  = t_MODE(_t, mega_t);
		//
		y[ty*Y + _y] = -FLT_MAX;
	}
}

void imax__pre_f(Inst_t * inst) {
	uint C0 = inst->params[0];
	FOR(0, mega_t, MEGA_T) {
		_FLT_MIN<<<dim3(KERD(inst->Y,16), KERD(GRAND_T,16)), dim3(16,16)>>>(
			inst->Y,
			inst->y__d,
			//
			mega_t,
			//
			C0
		);
	}
};

void imax__pre_batchique(Inst_t * inst) {
	
};