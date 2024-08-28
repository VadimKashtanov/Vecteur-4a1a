#include "matmul.cuh"

/*static __global__ void kerd__matmul__simple(
	uint x0_t, uint X0, float * x0,
	uint x1_t, uint X1, float * x1,
	//
	uint    Y,
	float * y,
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
		float s = 0;
		uint pos_y = ty*Y + _c0*(Bx*Ay) + _ay*(Bx) + _bx;
		FOR(0, k, Ax) {
			uint pos_x0 = tx0*C0*Ax*Ay + _c0*(Ax*Ay) + _ay*Ax + k;
			uint pos_x1 = tx1*C0*Bx*Ax + _c0*(Bx*Ax) + k*Bx + _bx;
			//
			if (x0[pos_x0] != x0[pos_x0] || x1[pos_x1]!=x1[pos_x1]) {
				printf("%f %f %i %i\n", x0[pos_x0], x1[pos_x1], pos_x0, pos_x1);
				assert(0);
			}
			s += x0[pos_x0] * x1[pos_x1];
		}
		y[pos_y] = s;
		assert(s==s);
	}
};*/

#define BLK 8
#define BLK_T 4

static __global__ void kerd__matmul__simple(
	uint x0_t, uint X0, float * x0,
	uint x1_t, uint X1, float * x1,
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
	__shared__ float B[BLK_T][BLK][BLK];

	//if (_bx < Bx && _ay < Ay && _t < GRAND_T) {
		uint tx0 = t_MODE(_t, mega_t-x0_t);
		uint tx1 = t_MODE(_t, mega_t-x1_t);
		uint ty  = t_MODE(_t, mega_t     );
		//
		float s = 0;
		//
		FOR(0, z, KERD(Ax,BLK)) {
			//
			uint _A_y = _ay;//(z*BLK_AX + thy);
			uint _A_x = (z*BLK + thx);
			A[thz][thy][thx] = ((_A_y<Ay && _A_x<Ax) ? x0[tx0*C0*Ax*Ay + _c0*(Ax*Ay) + _A_y*Ax + _A_x] : 0.0);//_ay*Ax + k];
			uint _B_y = (z*BLK + thy);
			uint _B_x = _bx;//(z*BLK_AX + thx);
			B[thz][thy][thx] = ((_B_y<Ax && _B_x<Bx) ? x1[tx1*C0*Bx*Ax + _c0*(Bx*Ax) + _B_y*Bx + _B_x] : 0.0);//k*Bx + _bx];
			
			//
			__syncthreads();
			//
			FOR(0, k, BLK) s += A[thz][thy][k] * B[thz][k][thx];
			__syncthreads();
		}
		//
	if (_bx < Bx && _ay < Ay && _t < GRAND_T)
		y[ty*Y + _c0*(Bx*Ay) + _ay*(Bx) + _bx] = s;
	//}
};

//	---------------------------------------------------------------------------------

void matmul__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t, uint entrainnement) {
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
		//FOR(0, _c0, C0) {
			//kerd__matmul__simple<<<dim3(KERD((Ay*C0),16), KERD((Bx*GRAND_T),16)), dim3(16,16)>>>(
			kerd__matmul__simple<<<dim3(KERD(Bx, BLK), KERD(Ay, BLK), KERD(GRAND_T*C0, BLK_T)), dim3(BLK,BLK,BLK_T)>>>(
				inst->x_t[0], inst->x_Y[0], x__d[0],
				inst->x_t[1], inst->x_Y[1], x__d[1],
				//
				inst->Y,
				inst->y__d,
				//
				ts__d, mega_t,
				//
			//	_c0,
				//
				Ax, Ay, Bx, C0
			);
		//}
	} else {
		inst_zero_mega_t(inst, mega_t);
	}
};