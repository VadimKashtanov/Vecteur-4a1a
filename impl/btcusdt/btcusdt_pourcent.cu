#include "btcusdt.cuh"

#include "../../impl_template/tmpl_etc.cu"

static __global__ void k__pourcent_btcusdt_stricte(
	float * somme,
	float * potentiel,
	//
	float * prixs,
	float * y, float * w,
	uint * ts__d,
	//
	float coef,
	//
	uint T, uint Y)
{
	uint _t = threadIdx.x + blockIdx.x * blockDim.x;
	//
	if (_t < GRAND_T) {
		FOR(0, mega_t, MEGA_T) {
			uint ty        = t_MODE(_t, mega_t);
			uint t_btcusdt = ts__d[_t] + mega_t;
			//
			assert(t_btcusdt != T-1);	//	Le dernier est chargé, mais est interdit, car le y__d est nulle (car on connait pas l'heure+1)
			//
			uint wpos = t_btcusdt*Y + 0; float _w = w[wpos];
			uint ypos = ty       *Y + 0; float _y = y[ypos];
			//
			float K = powf(100*fabs(prixs[t_btcusdt+1]/prixs[t_btcusdt] - 1), coef);
			//
			float a_t_il_predit = (float)(sng(_y) == sng(_w));
			//
			atomicAdd(&    somme[0], a_t_il_predit * K);
			atomicAdd(&potentiel[0],       1       * K);
		}
	}
};

float pourcent_btcusdt(BTCUSDT_t * btcusdt, float * y__d, uint * ts__d, float coef) {
	float *     somme__d = cudalloc<float>(1);
	float * potentiel__d = cudalloc<float>(1);
	//
	k__pourcent_btcusdt_stricte<<<dim3(KERD(GRAND_T, 16)), dim3(16)>>>(
		somme__d, potentiel__d,
		//
		btcusdt->prixs__d,
		y__d, btcusdt->y__d,
		ts__d,
		//
		coef,
		//
		btcusdt->T, btcusdt->Y
	);
	ATTENDRE_CUDA();
	//
	float * somme     = gpu_vers_cpu<float>(somme__d, 1);
	float * potentiel = gpu_vers_cpu<float>(potentiel__d, 1);
	float ret = somme[0] / potentiel[0];
	free(somme); free(potentiel);
	//
	return ret;
};
