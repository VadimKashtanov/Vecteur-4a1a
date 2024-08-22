#include "btcusdt.cuh"

#include "../../impl_template/tmpl_etc.cu"

static float * charger_gpu(FILE * fp, uint L) {
	float * l    = lire        <float>(fp, L);
	float * l__d = cpu_vers_gpu<float>(l,  L);
	free(l);
	return l__d;
}

BTCUSDT_t * cree_btcusdt(char * fichier) {
	//
	BTCUSDT_t * ret = (BTCUSDT_t*)malloc(sizeof(BTCUSDT_t));

	//
	FILE * fp = FOPEN(fichier, "rb");
	
	//
	FREAD(&ret->T,  sizeof(uint), 1, fp);
	FREAD(&ret->X,  sizeof(uint), 1, fp);
	FREAD(&ret->Y,  sizeof(uint), 1, fp);
	uint L,N;
	FREAD(&L,  sizeof(uint), 1, fp);
	FREAD(&N,  sizeof(uint), 1, fp);

	//
	ret->prixs__d = charger_gpu(fp, ret->T *    1  );
	ret->    x__d = charger_gpu(fp, ret->T * ret->X);
	ret->    y__d = charger_gpu(fp, ret->T * ret->Y);

	//
	fclose(fp);

	//
	printf("BTCUSDT : T=%i X=%i Y=%i\n", ret->T, ret->X, ret->Y);

	//
	return ret;
};

void liberer_btcusdt(BTCUSDT_t * donnee) {
	cudafree<float>(donnee->    y__d);
	cudafree<float>(donnee->    x__d);
	cudafree<float>(donnee->prixs__d);
};