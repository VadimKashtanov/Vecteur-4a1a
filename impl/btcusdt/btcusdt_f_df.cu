#include "btcusdt.cuh"

#include "../../impl_template/tmpl_etc.cu"

static __global__ void k__f_df_btcusdt(
	float * S,
	//
	float * prixs,
	//
	float * y, float * dy,
	float * w,
	uint * ts__d,
	//
	uint T, uint Y)
{
	uint _y = threadIdx.x + blockIdx.x * blockDim.x; 
	//uint _t = threadIdx.y + blockIdx.y * blockDim.y;
	//uint  i = threadIdx.z + blockIdx.z * blockDim.z;
	//
	if (_y < Y) {
		float s = 0;
		FOR(0, _t, GRAND_T) {
			FOR(0, mega_t, MEGA_T) {
				uint ty        = t_MODE(_t, mega_t);
				uint t_btcusdt = ts__d[_t] + 1 + mega_t;
				//
				assert(t_btcusdt != T-1);	//le dernier bloque n'a pas de y car on connait pas le future
				assert(t_btcusdt  < T  );	//verifier qu'il existe
				//
				uint wpos = t_btcusdt*Y + 0; float _w = w[wpos];
				uint ypos = ty       *Y + 0; float _y = y[ypos];
				//
				float K = powf(prixs[t_btcusdt+1]/prixs[t_btcusdt] - 1, P_COEF);
				//
				if (_y != _y) {
					printf("ypos=%i y=%f t_btcusdt=%i ty=%i T=%i dernier t=%i\n", ypos, _y, t_btcusdt, ty, T, T-1);
					assert(0);
				}
				assert(_y >= -100 && _y <= +100);
				//
				float coef = K / (float)(GRAND_T * MEGA_T * Y);
				s       += ( score_p2(_y, _w, 2)) * coef;
				float ds = (dscore_p2(_y, _w, 2)) * coef;
				//
				dy[ty*Y + ypos] = ds;
			}
		}
		//
		atomicAdd(&S[0], s);
	}
};

float f_df_btcusdt(BTCUSDT_t * btcusdt, float * y__d, float * dy__d, uint * ts__d) {
	float * S__d = cudalloc<float>(1);
	//
	k__f_df_btcusdt<<<dim3(KERD(btcusdt->Y, 32)), dim3(32)>>>(
		S__d,
		//
		btcusdt->prixs__d,
		//
		y__d, dy__d,
		btcusdt->y__d,
		ts__d,
		//
		btcusdt->T, btcusdt->Y
	);
	ATTENDRE_CUDA();
	//
	//
	float * S = gpu_vers_cpu<float>(S__d, 1);
	float s = S[0] / (float)(MEGA_T*GRAND_T*btcusdt->Y);
	//
	cudafree<float>(S__d);
	    free       (S   );
	//
	return s;
};