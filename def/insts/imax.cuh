#pragma once

#include "insts.cuh"

#define imax__Xs 1
#define imax__Ys 1
#define imax__PARAMS 1 //C0
#define imax__Nom "imax"

uint imax__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);
uint imax__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);

void imax__init_poids(Inst_t * inst);

void imax__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t, uint entrainnement);
void imax__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t);

void imax__pre_f(Inst_t * inst);
void imax__pre_batchique(Inst_t * inst);

static fonctions_insts_t fi_imax = {
	.Xs    =imax__Xs,
	.PARAMS=imax__PARAMS,
	.Nom   =imax__Nom,
	//
	.calculer_P=imax__calculer_P,
	.calculer_L=imax__calculer_L,
	//
	.init_poids=imax__init_poids,
	//
	.f =imax__f,
	.df=imax__df,
	//
	.pre_f=imax__pre_f,
	.pre_batchique=imax__pre_batchique
};