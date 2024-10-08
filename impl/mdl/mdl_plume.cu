#include "mdl.cuh"

#include "../impl_template/tmpl_etc.cu"

void plumer_model  (Mdl_t * mdl) {
	printf(" === Mdl_t : INSTS=%i ===\n", mdl->insts);
	//
	uint _max_xs = 0;
	FOR(0, i, mdl->insts) _max_xs = MAX2(_max_xs, inst_Xs[mdl->inst[i]->ID]);
	uint _max_params = 0;
	FOR(0, i, mdl->insts) _max_params = MAX2(_max_params, inst_PARAMS[mdl->inst[i]->ID]);
	//
	FOR(0, i, mdl->insts) {
		Inst_t * inst = mdl->inst[i];
		uint ID = inst->ID;
		//
		if (i % 2 == 0) printf("\033[2;1m");
		//
		printf("%3.i| ID=%3.i Y=%6.i P=%6.i L=%6.i : ", i, ID, inst->Y, inst->P, inst->L);

		printf("x_Y={");    FOR(0, j, inst_Xs[ID]    ) printf("%6.i,", inst->x_Y[j]  ); FOR(inst_Xs[ID], j, _max_xs) printf("%6.i,",0); printf("}, ");
		printf("x_pos={");  FOR(0, j, inst_Xs[ID]    ) printf("%3.i,", inst->x_pos[j]); FOR(inst_Xs[ID], j, _max_xs) printf("%3.i,",0); printf("}, ");
		printf("x_t={");    FOR(0, j, inst_Xs[ID]    ) printf("%i,",  inst->x_t[j]   ); FOR(inst_Xs[ID], j, _max_xs) printf("%i,",  0); printf("}, ");
		//
		printf("params={"); FOR(0, j, inst_PARAMS[ID]) printf("%4.i,", inst->params[j]);FOR(inst_PARAMS[ID], j, _max_params) printf("%4.i,", 0);printf("}, ");
		//
		printf(" inst=(%20s)\033[0m\n", inst_Nom[ID]);
	}

	//	Plumer l'ordre pour le shéma optimisé
	printf(" -- Optimisation --\n");
	FOR(0, i, mdl->BLOQUES) {
		printf("b=%i| ", i);
		FOR(0, j, mdl->elements[i]) printf("%i ", mdl->instructions[i][j]);
		printf("\n");
	};
};

void mdl_plume_poid(Mdl_t * mdl) {
	FOR(0, i, mdl->insts) {
		printf("###### %i-mem INST (ID=%i) #######\n", i, mdl->inst[i]->ID);
		float * p = gpu_vers_cpu<float>(mdl->inst[i]->p__d, mdl->inst[i]->P);
		FOR(0, j, mdl->inst[i]->P) printf("%i| %f\n", j, p[j]);
	};
};

void mdl_plume_grad(Mdl_t * mdl, BTCUSDT_t * btcusdt, uint * ts__d) {
	//
	mdl_allez_retour(mdl, btcusdt, ts__d);
	//
	printf("=== Grad ===\n");
	FOR(0, i, mdl->insts) {
		Inst_t * inst = mdl->inst[i];
		if (inst->P != 0) {
			//
			float *  p = gpu_vers_cpu<float>(inst-> p__d, inst->P);
			float * dp = gpu_vers_cpu<float>(inst->dp__d, inst->P);
			//
			//
			float pmax=p[0], pmin=p[0], pabsmax=fabs( p[0]), pmoyabs=fabs( p[0]);
			FOR(1, j, inst->P) {
				float val = p[j];
				if (val > pmax) pmax = val;
				if (val < pmin) pmin = val;
				if (fabs(val) > pabsmax) pabsmax = fabs(val);
				pmoyabs += fabs(val);
			}
			pmoyabs /= (float)inst->P;
			//
			//
			float gmax=dp[0], gmin=dp[0], gabsmax=fabs( dp[0]), gmoyabs=fabs( dp[0]);
			FOR(1, j, inst->P) {
				float val = dp[j];
				if (val > gmax) gmax = val;
				if (val < gmin) gmin = val;
				if (fabs(val) > gabsmax) gabsmax = fabs(val);
				gmoyabs += fabs(val);
			}
			gmoyabs /= (float)inst->P;
			//
			//
			free( p);
			free(dp);
			//
			//
			printf("%3.i| [ID=%2.i] Y=%5.i : Grad max=%+f; min=%+f absmax=%+f moy=%+f ;; Poid max=%+f; min=%+f absmax=%+f moy=%+f ;; Poids=%7.i\n",
				i, inst->ID, inst->Y,
				gmax, gmin, gabsmax, gmoyabs,
				pmax, pmin, pabsmax, pmoyabs,
				inst->P
			);
		};
	};
};