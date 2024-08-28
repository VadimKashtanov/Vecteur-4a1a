#include "matmul_poid_AP.cuh"

#define BLK 8
#define BLK_T 4

static __global__ void d_kerd__matmul_poid_AP__simple__dX(
	//dx = dY @ p.T
	uint x0_t, uint X0, float * x0, float * dx0,
	//
	float * p, float * dp,
	//
	uint    Y,
	float * y, float * dy,
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
	uint _ax = threadIdx.x + blockIdx.x * blockDim.x;
	uint _ay = threadIdx.y + blockIdx.y * blockDim.y;
	uint _tc0  = threadIdx.z + blockIdx.z * blockDim.z;
	//
	uint _t  = _tc0 % GRAND_T;
	uint _c0 = (_tc0-_t)/GRAND_T;
	//
	__shared__ float DY[BLK_T][BLK][BLK];
	__shared__ float Pt       [BLK][BLK];
	//
	uint tx0 = t_MODE(_t, mega_t-x0_t);
	uint ty  = t_MODE(_t, mega_t     );
	//
	float s = 0;
	FOR(0, z, KERD(Bx, BLK)) {
		uint _DY_x = z*BLK + thx;//_ax;
		uint _DY_y = _ay;//z*BLK + thx;
		DY[thz][thy][thx] = ((_DY_x<Bx && _DY_y<Ay) ? dy[ty*C0*Bx*Ay + _c0*Bx*Ay + _DY_y*Bx+_DY_x]:0);
		uint _Pt_x = _ax;//z*BLK + thy;
		uint _Pt_y = z*BLK + thy;//_bx;
		if (thz == 0)
			Pt[thy][thx] = ((_Pt_x<Ax && _Pt_y<Bx) ? p[_c0*Bx*Ax + TRANSPOSE(_Pt_x,_Pt_y,Ax,Bx)]:0);
		__syncthreads();

		FOR(0, k, BLK) s += DY[thz][thy][k] * Pt[k][thx];
		__syncthreads();
	};

	if (_ax < Ax && _ay < Ay) {
		atomicAdd(&dx0[tx0*C0*Ax*Ay + _c0*(Ax*Ay) + _ay*Ax + _ax], s);
	}
};

static __global__ void d_kerd__matmul_poid_AP__simple__dP(
	//dp = x.T @ dY
	uint x0_t, uint X0, float * x0, float * dx0,
	//
	float * p, float * dp,
	//
	uint    Y,
	float * y, float * dy,
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
	uint _ax = threadIdx.y + blockIdx.y * blockDim.y;
	uint _tc0  = threadIdx.z + blockIdx.z * blockDim.z;
	//
	uint _t  = _tc0 % GRAND_T;
	uint _c0 = (_tc0-_t)/GRAND_T;
	//
	__shared__ float Xt[BLK_T][BLK][BLK];
	__shared__ float DY[BLK_T][BLK][BLK];
	//
	uint tx0 = t_MODE(_t, mega_t-x0_t);
	uint ty  = t_MODE(_t, mega_t     );
	//
	float s = 0;
	FOR(0, z, KERD(Ay,BLK)) {
		uint _Xt_x = _ax;
		uint _Xt_y = z*BLK + thx;
		Xt[thz][thy][thx] = ((_Xt_x<Ax && _Xt_y<Ay) ? x0[tx0*C0*Ax*Ay + _c0*Ax*Ay + _Xt_y*Ax+_Xt_x/*TRANSPOSE(_Xt_x,_Xt_y, Ax,Ay)*/]:0);
		uint _DY_x = _bx;//z*BLK + thy;
		uint _DY_y = z*BLK + thy;//_bx;
		DY[thz][thy][thx] = ((_DY_x<Bx && _DY_y<Ay) ? dy[ty *C0*Bx*Ay + _c0*Bx*Ay + _DY_y*Bx + _DY_x]:0);
		__syncthreads();

		FOR(0, k, BLK) s += Xt[thz][thy][k] * DY[thz][k][thx];
		__syncthreads();
	};

	if (_ax < Ax && _bx < Bx) {
		atomicAdd(&dp[_c0*(Bx*Ax) + _ax*Bx + _bx], s);
	}
};

//	---------------------------------------------------------------------------------

void matmul_poid_AP__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t) {
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
		//FOR(0, _c0, C0) {
			d_kerd__matmul_poid_AP__simple__dX<<<dim3(KERD(Ax, BLK), KERD(Ay, BLK), KERD(GRAND_T*C0, BLK_T)), dim3(BLK,BLK,BLK_T)>>>(
				inst->x_t[0], inst->x_Y[0], x__d[0], dx__d[0],
				//
				inst->p__d, inst->dp__d,
				//
				inst->Y,
				inst->y__d, inst->dy__d,
				//
				ts__d, mega_t,
				//
			//	_c0,
				//
				Ax, Ay, Bx, C0
			);
			d_kerd__matmul_poid_AP__simple__dP<<<dim3(KERD(Bx, BLK), KERD(Ax, BLK), KERD(GRAND_T*C0, BLK_T)), dim3(BLK,BLK,BLK_T)>>>(
				inst->x_t[0], inst->x_Y[0], x__d[0], dx__d[0],
				//
				inst->p__d, inst->dp__d,
				//
				inst->Y,
				inst->y__d, inst->dy__d,
				//
				ts__d, mega_t,
				//
			//	_c0,
				//
				Ax, Ay, Bx, C0
			);
		//}
	} else {
		//inst_zero_mega_t(inst, mega_t);
	}
};