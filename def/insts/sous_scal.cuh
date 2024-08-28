#pragma once

#include "insts.cuh"

#define sous_scal__Xs 2
#define sous_scal__Ys 1
#define sous_scal__PARAMS 1 //C0
#define sous_scal__Nom "sous_scal"

uint sous_scal__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);
uint sous_scal__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);

void sous_scal__init_poids(Inst_t * inst);

void sous_scal__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t, uint entrainnement);
void sous_scal__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t);

void sous_scal__pre_f(Inst_t * inst);
void sous_scal__pre_batchique(Inst_t * inst);

static fonctions_insts_t fi_sous_scal = {
	.Xs    =sous_scal__Xs,
	.PARAMS=sous_scal__PARAMS,
	.Nom   =sous_scal__Nom,
	//
	.calculer_P=sous_scal__calculer_P,
	.calculer_L=sous_scal__calculer_L,
	//
	.init_poids=sous_scal__init_poids,
	//
	.f =sous_scal__f,
	.df=sous_scal__df,
	//
	.pre_f=sous_scal__pre_f,
	.pre_batchique=sous_scal__pre_batchique
};