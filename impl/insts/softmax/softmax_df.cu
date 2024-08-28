#include "softmax.cuh"

#define _VECT_ 128

__global__
static void d_kerd__softmax(
	uint x0_t, uint X0, float * x0, float * dx0,
	//
	uint    Y,
	float * y, float * dy,
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
	/*float * _x_ = shared_dynamique + 0;
	float * max = shared_dynamique + Vect;
	float * sum = shared_dynamique + Vect + Vect;
	//
	float * d_x_ = shared_dynamique + Vect + Vect + Vect;
	float * dmax = shared_dynamique + Vect + Vect + Vect + Vect;
	float * dsum = shared_dynamique + Vect + Vect + Vect + Vect + 1;*/
	float * __y_ = shared_dynamique + 0;
	float * _dy_ = shared_dynamique + Vect;
	float * d_x_ = shared_dynamique + Vect + Vect;
	/*//__shared__ float __x_[_VECT_];
	__shared__ float __y_[_VECT_];
	__shared__ float _dy_[_VECT_];
	//__shared__ float max[_VECT_];
	//__shared__ float sum[_VECT_];
	//
	__shared__ float d_x_[_VECT_];
	//__shared__ float dmax[1];
	//__shared__ float dsum[1];*/
	//
	//
	//__x_[thx] = x0[tx0*X0 + _v*Vect + thx];
	__y_[thx] =  y[ty *Y  + _v*Vect + thx];
	_dy_[thx] = dy[ty *Y  + _v*Vect + thx];
	d_x_[thx] = 0;
	//if (thx==0) dmax[thx] = 0;
	//if (thx==0) dsum[thx] = 0;
	__syncthreads();

	FOR(0, k, _VECT_) {
		d_x_[thx] += __y_[k] * ((k==thx) - __y_[thx]) * _dy_[k];
		__syncthreads();
	}

	atomicAdd(&dx0[tx0*X0 + _v*Vect + thx], d_x_[thx]);

	//

	/*{
		//	1) Max
		max[thx] = _x_[thx];
		__syncthreads();
		//
		//atomicMax(&max[0], _x_[thx]);
		uint lg = (uint)log2f((float)Vect);
		FOR(0, l, lg) {
			//uint p = pow(2, l);
			uint p = 1 << l;
			if (thx % p == 0) max[thx] = MAX2(max[thx], max[thx+p]);
			__syncthreads();
		}
		__syncthreads();
	}

	float val;
	{
		//	2) exp(x-max) && sum
		val = expf(_x_[thx] - max[0]);
		sum[thx] = val;
		__syncthreads();
		//
		uint lg = (uint)log2f((float)Vect);
		FOR(0, l, lg) {
			//uint p = pow(2, l);
			uint p = 1 << l;
			if (thx % p == 0) sum[thx] = sum[thx] + sum[thx+p];
			__syncthreads();
		}
		__syncthreads();
	}

	{
		//	3) /= sum
		assert(sum[0] == sum[0]);
		val /= sum[0];
		assert(val == val);
		//
		//float _val = _x_[thx] / sum[0];
		//float val = y[ty*X0 + _v*Vect + thx];
		float _dy = dy[ty*X0 + _v*Vect + thx];
		float _x__thx = val * sum[0];
		printf("%f %f %f\n", _dy, sum[0], _x__thx);
		d_x_[thx] += _dy * 1 / sum[0];
		float ____ = _dy * (-1)*_x__thx / (sum[0]*sum[0]);
		if (____ != ____) {
			printf("%i %i %f\n", thx, _v, ____);
			assert(0);
		}
		atomicAdd(&dsum[0], ____);
		__syncthreads();
	}

	{
		//	4) deriv : 3) exp(x-max) && sum
		val *= sum[0];
		assert(val == val);
		assert(dsum[0] == dsum[0]);
		//sum[thx] = val;
		//float dval = dsum[0];
		d_x_[thx] += dsum[0] * val * 1;
		atomicAdd(&dmax[ 0 ], dsum[0] * val * (-1));
		__syncthreads();
	}
	
	{
		if (val==max[0]) d_x_[thx] += dmax[0];
		__syncthreads();
	}

	if (d_x_[thx] != d_x_[thx]) {
		assert(dmax[0]==dmax[0]);
		assert(dsum[0]==dsum[0]);
		printf("thx=%i _v=%i d_x_[thx]=%f\n", thx, _v, d_x_[thx]);
		assert(0);
	};
	atomicAdd(&dx0[tx0*X0 + _v*Vect + thx], d_x_[thx]);*/
};

void softmax__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t) {
	uint X0 = inst->x_Y[0]; uint x0_t = inst->x_t[0];
	//
	uint \
		Vect    = inst->params[0];
	//
	bool x0_existe = (mega_t != 0 ? true : (x0_t != 1));
	//
	uint Vects = X0/Vect;
	//
	if (x0_existe) {
		d_kerd__softmax<<<dim3(Vects, GRAND_T), dim3(Vect,1), (3*Vect)*sizeof(float)>>>(
			inst->x_t[0], inst->x_Y[0], x__d[0], dx__d[0],
			//
			inst->Y,
			inst->y__d, inst->dy__d,
			//
			mega_t,
			//
			Vect
		);
	} else {
		//	rien
	}
};