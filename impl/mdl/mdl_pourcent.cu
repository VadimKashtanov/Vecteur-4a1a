#include "mdl.cuh"

float mdl_pourcent(Mdl_t * mdl, BTCUSDT_t * btcusdt, uint * ts__d, float coef) {
	mdl_f(mdl, btcusdt, ts__d, false);
	return pourcent_btcusdt(btcusdt, mdl->inst[mdl->sortie]->y__d, ts__d, coef);
};