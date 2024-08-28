#include "matmul.cuh"

/*static __global__ void d_kerd__matmul__simple(
	uint x0_t, uint X0, float * x0, float * dx0,
	uint x1_t, uint X1, float * x1, float * dx1,
	//
	uint    Y,
	float * y, float * dy,
	//
	uint * ts__d, uint mega_t,
	//
	uint Ax, uint Ay, uint Bx, uint C0)
{
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;
	uint thy = threadIdx.y + blockIdx.y * blockDim.y;

	//	thx = Ay*C0
	uint _ay = thx % Ay;
	uint _c0 = (thx-_ay)/Ay;

	//	thy = Bx*GRAND_T
	uint _bx = thy % Bx;
	uint  _t = (thy-_bx)/Bx;

	if (_ay < Ay && _c0 < C0 && _t < GRAND_T) {
		uint tx0 = t_MODE(_t, mega_t-x0_t);
		uint tx1 = t_MODE(_t, mega_t-x1_t);
		uint ty  = t_MODE(_t, mega_t     );
		//
		uint pos_y = ty*Y + _c0*(Bx*Ay) + _ay*(Bx) + _bx;
		float _dy = dy[pos_y];
		//
		FOR(0, k, Ax) {
			uint pos_x0 = tx0*C0*Ax*Ay + _c0*(Ax*Ay) + _ay*Ax + k;
			uint pos_x1 = tx1*C0*Bx*Ax + _c0*(Bx*Ax) + k*Bx + _bx;
			//
			//s += x0[pos_x0] * x1[pos_x0];
			atomicAdd(&dx0[pos_x0], x1[pos_x1] * _dy);
			atomicAdd(&dx1[pos_x1], x0[pos_x0] * _dy);
		}
	}
};*/

#define BLK 8
#define BLK_T 4

static __global__ void d_kerd__matmul__simple__dA(
	//dx = dY @ p.T
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
	__shared__ float Bt[BLK_T][BLK][BLK];
	//
	uint tx0 = t_MODE(_t, mega_t-x0_t);
	uint tx1 = t_MODE(_t, mega_t-x1_t);
	uint ty  = t_MODE(_t, mega_t     );
	//
	float s = 0;
	FOR(0, z, KERD(Bx, BLK)) {
		uint _DY_x = z*BLK + thx;//_ax;
		uint _DY_y = _ay;//z*BLK + thx;
		DY[thz][thy][thx] = ((_DY_x<Bx && _DY_y<Ay) ? dy[ty*C0*Bx*Ay + _c0*Bx*Ay + _DY_y*Bx+_DY_x]:0);
		uint _Bt_x = _ax;//z*BLK + thy;
		uint _Bt_y = z*BLK + thy;//_bx;
		Bt[thz][thy][thx] = ((_Bt_x<Ax && _Bt_y<Bx) ? x1[tx1*C0*Bx*Ax + _c0*Bx*Ax + TRANSPOSE(_Bt_x,_Bt_y,Ax,Bx)]:0);
		__syncthreads();

		FOR(0, k, BLK) s += DY[thz][thy][k] * Bt[thz][k][thx];
		__syncthreads();
	};

	if (_ax < Ax && _ay < Ay) {
		atomicAdd(&dx0[tx0*C0*Ax*Ay + _c0*(Ax*Ay) + _ay*Ax + _ax], s);
	}
};

static __global__ void d_kerd__matmul__simple__dB(
	//dp = x.T @ dY
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
	__shared__ float At[BLK_T][BLK][BLK];
	__shared__ float DY[BLK_T][BLK][BLK];
	//
	uint tx0 = t_MODE(_t, mega_t-x0_t);
	uint tx1 = t_MODE(_t, mega_t-x1_t);
	uint ty  = t_MODE(_t, mega_t     );
	//
	float s = 0;
	FOR(0, z, KERD(Ay,BLK)) {
		uint _At_x = _ax;
		uint _At_y = z*BLK + thx;
		At[thz][thy][thx] = ((_At_x<Ax && _At_y<Ay) ? x0[tx0*C0*Ax*Ay + _c0*Ax*Ay + _At_y*Ax+_At_x/*TRANSPOSE(_Xt_x,_Xt_y, Ax,Ay)*/]:0);
		uint _DY_x = _bx;//z*BLK + thy;
		uint _DY_y = z*BLK + thy;//_bx;
		DY[thz][thy][thx] = ((_DY_x<Bx && _DY_y<Ay) ? dy[ty *C0*Bx*Ay + _c0*Bx*Ay + _DY_y*Bx + _DY_x]:0);
		__syncthreads();

		FOR(0, k, BLK) s += At[thz][thy][k] * DY[thz][k][thx];
		__syncthreads();
	};

	if (_ax < Ax && _bx < Bx && _t < GRAND_T) {
		atomicAdd(&dx1[tx1*C0*Bx*Ax + _c0*(Bx*Ax) + _ax*Bx + _bx], s);
	}
};

//	---------------------------------------------------------------------------------

void matmul__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t) {
	uint * params = inst->params;
	uint \
		Ax =params[0],	\
		Ay =params[1],	\
		Bx =params[2],	\
		C0 =params[3];
	//
	uint x0_t = inst->x_t[0];
	uint x1_t = inst->x_t[1];
	//
	bool x0_existe = (mega_t != 0 ? true : (x0_t != 1));
	bool x1_existe = (mega_t != 0 ? true : (x1_t != 1));
	//
	ASSERT(x0_existe && x1_existe);
	//
	if (x0_existe && x1_existe) {
		/*d_kerd__matmul__simple<<<dim3(KERD((Ay*C0),16), KERD((Bx*GRAND_T),16)), dim3(16,16)>>>(
			inst->x_t[0], inst->x_Y[0], x__d[0], dx__d[0],
			inst->x_t[1], inst->x_Y[1], x__d[1], dx__d[1],
			//
			inst->Y,
			inst->y__d, inst->dy__d,
			//
			ts__d, mega_t,
			//
			Ax, Ay, Bx, C0
		);*/


			d_kerd__matmul__simple__dA<<<dim3(KERD(Ax, BLK), KERD(Ay, BLK), KERD(GRAND_T*C0, BLK_T)), dim3(BLK,BLK,BLK_T)>>>(
				inst->x_t[0], inst->x_Y[0], x__d[0], dx__d[0],
				inst->x_t[1], inst->x_Y[1], x__d[1], dx__d[1],
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
			d_kerd__matmul__simple__dB<<<dim3(KERD(Bx, BLK), KERD(Ax, BLK), KERD(GRAND_T*C0, BLK_T)), dim3(BLK,BLK,BLK_T)>>>(
				inst->x_t[0], inst->x_Y[0], x__d[0], dx__d[0],
				inst->x_t[1], inst->x_Y[1], x__d[1], dx__d[1],
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
	} else {
		//inst_zero_mega_t(inst, mega_t);
	}
};