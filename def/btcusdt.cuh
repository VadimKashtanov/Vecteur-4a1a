#pragma once

#include "meta.cuh"

#define  score_p2(y,w,C) (powf(y-w, C  )/(float)C )
#define dscore_p2(y,w,C) (powf(y-w, C-1)          )

//	------------------------------------

void btcusdt_csv(char * csv, char * dar);

//	------------------------------------

#define P_COEF 1.0

typedef struct {
	//
	uint T;	//T - 1 (car le dernier ne connais pas son prochain y)

	uint X; uint Y;

	//	Espaces
	float * prixs__d;	//	T
	float *     x__d;	//	T * (I * N * L)
	float *     y__d;	//	T * (I * 1 * L)
} BTCUSDT_t;

BTCUSDT_t * cree_btcusdt(char * fichier);
void     liberer_btcusdt(BTCUSDT_t * btcusdt);
//
float pourcent_btcusdt(BTCUSDT_t * btcusdt, float * y__d, uint * ts__d, float coef);
//
float f_df_btcusdt(BTCUSDT_t * btcusdt, float * y__d, float * dy__d, uint * ts__d);