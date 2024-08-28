#include "sous_scal.cuh"

uint sous_scal__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	return 0;
};

uint sous_scal__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	return 0;
};

void sous_scal__init_poids(Inst_t * inst) {
	ASSERT(inst->Y == inst->x_Y[0]);
	uint C0 = inst->params[0];
	ASSERT(inst->x_Y[0] % inst->params[0] == 0);
	ASSERT(inst->x_Y[1] == C0);
	//inst->p__d;
};

void sous_scal__pre_f(Inst_t * inst) {

};

void sous_scal__pre_batchique(Inst_t * inst) {
	
};