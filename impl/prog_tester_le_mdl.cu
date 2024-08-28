#include "main.cuh"

#include "../impl_template/tmpl_etc.cu"

static float * toutes_les_predictions_T_plus_que_GRAND_T(Mdl_t * mdl, BTCUSDT_t * btcusdt) {
	uint T = btcusdt->T;
	uint Y = mdl->inst[mdl->sortie]->Y;
	//
	uint gm  = GRAND_T*MEGA_T;
	uint _T_ = (T%gm==0 ?     T/gm   :   (T - (T%gm) + gm) / gm   );
	//
	float * les_predictions = alloc<float>(T*Y);
	//
	FOR(0, _t_, _T_) {
		//
		uint ts[GRAND_T];
		FOR(0, t, GRAND_T) ts[t] = (_t_*gm + t*MEGA_T) % T;
		uint * ts__d = cpu_vers_gpu<uint>(ts, GRAND_T);
		
		//
		mdl_f(mdl, btcusdt, ts__d, false);

		//
		uint Y    = mdl->inst[mdl->sortie]->Y;
		float * y = gpu_vers_cpu<float>(mdl->inst[mdl->sortie]->y__d, GRAND_T*MEGA_T*Y);
		
		//
		FOR(0, grand_t, GRAND_T) {
			FOR(0, mega_t, MEGA_T) {
				if (_t_*gm + grand_t*MEGA_T + mega_t < T) {
					uint ty = t_MODE(grand_t, mega_t);
					//
					FOR(0, l, Y) les_predictions[(_t_*gm + grand_t*MEGA_T + mega_t)*Y + l] = y[ty*Y + l];
				}
			}
		}

		//
		cudafree<uint>(ts__d);
		free(y);
	};
	//
	return les_predictions;
};

static float * toutes_les_predictions_T_moins_que_GRAND_T(Mdl_t * mdl, BTCUSDT_t * btcusdt) {
	uint T = btcusdt->T;
	uint Y = mdl->inst[mdl->sortie]->Y;
	//
	float * les_predictions = alloc<float>(T*Y);
	//
	uint ts[GRAND_T];
	FOR(0, t, GRAND_T) ts[t] = t % btcusdt->T;
	uint * ts__d = cpu_vers_gpu<uint>(ts, GRAND_T);
	
	//
	mdl_f(mdl, btcusdt, ts__d, false);

	//
	float * y = gpu_vers_cpu<float>(mdl->inst[mdl->sortie]->y__d, GRAND_T*MEGA_T*Y);
		
	//
	FOR(0, t, btcusdt->T) {
		uint ty = t_MODE(t, 0);
		//
		FOR(0, l, Y) les_predictions[t*Y + l] = y[ty*Y + l];
	}

	//
	cudafree<uint>(ts__d);
	free(y);
	//
	return les_predictions;
};

int main(int argc, char ** argv) {
	//	=========================================================
	srand(0);
	init_listes_instructions();
	ecrire_structure_generale("structure_generale.bin");
	verif_insts();

	//	=========================================================
	BTCUSDT_t * btcusdt = cree_btcusdt("dar_tester_le_model.bin");

	//	=========================================================
	//	--- Mdl_t ---
	Mdl_t * mdl = ouvrire_mdl("mdl.bin");
	//
	float * preds;
	if (MEGA_T == 1) {
		if      (btcusdt->T >= GRAND_T) preds = toutes_les_predictions_T_plus_que_GRAND_T(mdl, btcusdt);
		else if (btcusdt->T <  GRAND_T) preds = toutes_les_predictions_T_moins_que_GRAND_T(mdl, btcusdt);
	} else {
		ASSERT(btcusdt->T % (GRAND_T*MEGA_T));
		preds = toutes_les_predictions_T_plus_que_GRAND_T(mdl, btcusdt);
	}

	//
	float * prixs = gpu_vers_cpu<float>(btcusdt->prixs__d, btcusdt->T);

	//
	uint Y = mdl->inst[mdl->sortie]->Y;

	FILE * fp = FOPEN("les_predictions.bin", "wb");
	//
	FWRITE(&btcusdt->T, sizeof(uint),  1,            fp);
	FWRITE(&Y,          sizeof(uint),  1,            fp);
	FWRITE(prixs,       sizeof(float), btcusdt->T,   fp);
	FWRITE(preds,       sizeof(float), btcusdt->T*Y, fp);
	//
	fclose(fp);

	//
	free(preds);
	free(prixs);
	liberer_mdl    (mdl    );
	liberer_btcusdt(btcusdt);
};