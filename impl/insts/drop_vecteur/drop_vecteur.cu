#include "drop_vecteur.cuh"

#include "../../impl_template/tmpl_etc.cu"

uint drop_vecteur__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	return 0;
};

uint drop_vecteur__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	return 0;
};

void drop_vecteur__init_poids(Inst_t * inst) {
	uint VECT = inst->params[0];
	uint POURCENT = inst->params[1];
	//
	ASSERT(inst->Y == inst->x_Y[0]);
	ASSERT(VECT > 0);
	ASSERT(POURCENT < 100);
	ASSERT(inst->x_Y[0] % VECT == 0);
};

void drop_vecteur__pre_f(Inst_t * inst) {
	//uint * a  = (uint*)malloc(sizeof(uint)*1);
	//a[0] = (rand() % 10000);
	//inst->espace_potentiel = a;
};

void drop_vecteur__pre_batchique(Inst_t * inst) {
	if (inst->espace_potentiel != 0) cudafree<float>((float*)inst->espace_potentiel);
	float * matrice = (float*)malloc(sizeof(float) * inst->Y);
	FOR(0, i, inst->Y) {
		if ((rand()%100) < inst->params[1]) matrice[i] = 0;
		else                                matrice[i] = 1;
	}
	inst->espace_potentiel = (void*)cpu_vers_gpu<float>(matrice, inst->Y);
	free(matrice);
};