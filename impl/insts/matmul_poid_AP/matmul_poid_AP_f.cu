#include "matmul_poid_AP.cuh"

#define BLK 8
#define BLK_T 4

static __global__ void kerd__matmul_poid_AP__simple(
	uint x0_t, uint X0, float * x0,
	//
	float * p,
	//
	uint    Y,
	float * y,
	//
	uint * ts__d, uint mega_t,
	//
	//uint _c0,
	//
	uint Ax, uint Ay, uint Bx, uint C0)
{
	uint thx = threadIdx.x;
	uint thy = threadIdx.y;
	uint thz = threadIdx.z;
	//
	uint _bx = threadIdx.x + blockIdx.x * blockDim.x;
	uint _ay = threadIdx.y + blockIdx.y * blockDim.y;
	uint _tc0  = threadIdx.z + blockIdx.z * blockDim.z;

	uint _t  = _tc0 % GRAND_T;
	uint _c0 = (_tc0-_t)/GRAND_T;

	__shared__ float A[BLK_T][BLK][BLK];
	__shared__ float P       [BLK][BLK];

	//if (_bx < Bx && _ay < Ay && _t < GRAND_T) {
		uint tx0 = t_MODE(_t, mega_t-x0_t);
		uint ty  = t_MODE(_t, mega_t     );
		//
		float s = 0;
		//
		FOR(0, z, KERD(Ax,BLK)) {
			//
			uint _A_y = _ay;//(z*BLK_AX + thy);
			uint _A_x = (z*BLK + thx);
			A[thz][thy][thx] = ((_A_y<Ay && _A_x<Ax) ? x0[tx0*C0*Ax*Ay + _c0*(Ax*Ay) + _A_y*Ax + _A_x] : 0.0);//_ay*Ax + k];
			if (thz==0) {
				uint _P_y = (z*BLK + thy);
				uint _P_x = _bx;//(z*BLK_AX + thx);
				P[thy][thx] = ((_P_y<Ax && _P_x<Bx) ? p[_c0*(Bx*Ax) + _P_y*Bx + _P_x] : 0.0);//k*Bx + _bx];
			}
			//
			__syncthreads();
			//
			FOR(0, k, BLK) s += A[thz][thy][k] * P[k][thx];
			__syncthreads();
		}
		//
	if (_bx < Bx && _ay < Ay && _t < GRAND_T)
		y[ty*Y + _c0*(Bx*Ay) + _ay*(Bx) + _bx] = s;
	//}
};

//	---------------------------------------------------------------------------------

void matmul_poid_AP__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t, uint entrainnement) {
	uint * params = inst->params;
	uint \
		Ax =params[0],	\
		Ay =params[1],	\
		Bx =params[2],	\
		C0 =params[3];
	//
	uint x0_t = inst->x_t[0];
	//
	bool x0_existe = (mega_t != 0 ? true : (x0_t != 1));
	//
	//ASSERT(x0_existe);
	//
	if (x0_existe) {
		//FOR(0, c0, C0) {
			kerd__matmul_poid_AP__simple<<<dim3(KERD(Bx, BLK), KERD(Ay, BLK), KERD(GRAND_T*C0, BLK_T)), dim3(BLK,BLK,BLK_T)>>>(
				inst->x_t[0], inst->x_Y[0], x__d[0],
				//
				inst->p__d,
				//
				inst->Y,
				inst->y__d,
				//
				ts__d, mega_t,
				//
		//		c0,
				//
				Ax, Ay, Bx, C0
			);
		//}
	} else {
		inst_zero_mega_t(inst, mega_t);
	}
};