#include "softmax.cuh"

#define _VECT_ 128

__global__
static void kerd__softmax(
	uint x0_t, uint X0, float * x0,
	//
	uint    Y,
	float * y,
	//
	uint mega_t,
	//
	uint Vect)
{
	extern __shared__ float shared_dynamique[];
	assert(shared_dynamique != 0);
	//
	//	softmax(x) = exp(x-max) / sum(exp(x-max))
	//	1 max
	//	2 exp(x-max) && sum
	//	3 /= sum
	//
	uint thx = threadIdx.x;
	//
	uint _v = blockIdx.x;
	uint _t = blockIdx.y;
	//
	uint tx0 = t_MODE(_t, mega_t-x0_t);
	uint ty  = t_MODE(_t, mega_t     );
	//
	//
	float * _x_ = shared_dynamique + 0;
	//float * max = shared_dynamique + Vect;
	//float * sum = shared_dynamique + Vect + Vect;
	//__shared__ float _x_[_VECT_];
	//__shared__ float max[_VECT_];
	//__shared__ float sum[_VECT_];
	//
	//
	_x_[thx] = x0[tx0*X0 + _v*Vect + thx];
	__syncthreads();
	//
	//
	//
	//	1) Max
	//max[thx] = _x_[thx];
	//__syncthreads();
	//
	//atomicMax(&max[0], _x_[thx]);
	/*uint lg = (uint)log2f((float)Vect);
	FOR(1, l, lg+1) {
		//uint p = pow(2, l);
		uint p = 1 << l;
		uint p1 = 1 << (l-1);
		if (thx % p == 0) max[thx] = MAX2(max[thx], max[thx+p1]);
		__syncthreads();
	}
	__syncthreads();*/
	//
	//
	//
	//	2) exp(x-max) && sum
	float val = expf(_x_[thx]);//expf(_x_[thx] - max[0]);
	//if (_t==0 && _v==0) printf("%f, ", val);
	//_x_[thx] = val;
#define sum _x_
	sum[thx] = val;
	__syncthreads();
	//
	uint lg = (uint)log2f((float)Vect);
	FOR(1, l, lg+1) {
		//uint p = pow(2, l);
		uint p = 1 << l;
		uint p1 = 1 << (l-1);
		if (thx % p == 0) {
			//printf("%f = %f+%f\n", sum[thx] + sum[thx+p], sum[thx], sum[thx+p]);
			sum[thx] = sum[thx] + sum[thx+p1];
		}
		__syncthreads();
	}
	__syncthreads();
	//if (_t==0 && _v==0 && thx==0) printf("\nsomme = %f\n", _x_[0]);
	//
	//
	//
	//	3) /= sum
	assert(sum[0] == sum[0]);
	//FOR(0, j, 128) printf("somme=%f ", sum[j]);
	//printf("\n");
	//
	val = val / _x_[0];
	y[ty*X0 + _v*Vect + thx] = val;
	assert(val == val);
	//printf("%f\n", val);
	assert(val <= 1 && val >= 0);
};

void softmax__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t, uint entrainnement) {
	uint X0 = inst->x_Y[0]; uint x0_t = inst->x_t[0];
	//
	uint \
		Vect = inst->params[0];
	//
	ASSERT(Vect < 1024);
	ASSERT((Vect & (Vect - 1)) == 0);	//Vect est une puissance de 2
	//
	ASSERT(Vect == _VECT_);
	//
	bool x0_existe = (mega_t != 0 ? true : (x0_t != 1));
	//
	uint Vects = X0 / Vect;
	//
	if (x0_existe) {
		kerd__softmax<<<dim3(Vects, GRAND_T), dim3(Vect,1), (Vect*sizeof(float) + Vect*sizeof(float) + Vect*sizeof(float))>>>(
			inst->x_t[0], inst->x_Y[0], x__d[0],
			//
			inst->Y,
			inst->y__d,
			//
			mega_t,
			//
			Vect
		);
	} else {
		inst_zero_mega_t(inst, mega_t);
	}
};