#include "mul.cuh"

uint mul__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	return 0;
};

uint mul__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	return 0;
};

void mul__init_poids(Inst_t * inst) {
	ASSERT(inst->Y == inst->x_Y[0] && inst->Y == inst->x_Y[1]);
};

void mul__pre_f(Inst_t * inst) {
	
};

void mul__pre_batchique(Inst_t * inst) {
	
};