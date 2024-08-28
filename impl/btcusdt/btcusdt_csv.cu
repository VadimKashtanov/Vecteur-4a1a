#include "btcusdt.cuh"

#include "../../impl_template/tmpl_etc.cu"

static char * lire_chars(FILE * fp) {
	fseek(fp, 0, SEEK_END);
	uint len = ftell(fp);
	ASSERT(len != 0);
	fseek(fp, 0, SEEK_SET);
	char * str = (char*)malloc(len+1);
	FREAD(str, 1, len, fp);
	str[len-1+1] = '\0';
	return str;
};

static uint nombre_lignes(char * txt) {
	uint _lignes = 0;
	uint pos = 0;
	while ( txt[pos] != '\0') {
		if (txt[pos] == '\n') _lignes++;
		pos++;
	}
	return _lignes+1; // car la dernière se termine avec un '\0'
};

static char *** tableau_csv(char * txt) {
	//	-- Parser le CSV --
	uint lignes = nombre_lignes(txt);
	char ** ligne = split(txt, '\n');//texte_vers_lignes(txt, lignes);
	//FOR(0, i, lignes) printf("%s\n", ligne[i]);

	//	-- Traiter le CSV --
	free(ligne[0]); free(ligne[1]);	//https ...\n Hight,low,close...
	lignes -= 2;
	FOR(0, i, lignes) ligne[i] = ligne[i+2];
	//
	char *** tableau = (char***)malloc(sizeof(char**) * lignes); //lignes_vers_tableau(ligne, lignes);
	FOR(0, i, lignes) {
		tableau[i] = split(ligne[i], ',');
		//FOR(0, m, MOTS) printf("[%s]", tableau[i][m]);
		//printf("\n");
		free(ligne[i]);
	}
	free(ligne);
	//
	return tableau;
};

//	--------------------------------------------------

typedef struct {
	char  *  nom;
	float * ligne;
	uint T;
} Ligne_t;

static Ligne_t * extraire(char * nom, char *** tableau, uint T, uint mot) {
	Ligne_t * ligne = alloc<Ligne_t>(1);
	ligne->nom = nom;
	ligne->ligne = alloc<float>(T);
	ligne->T = T;
	FOR(0, i, T) ligne->ligne[i] = atof(tableau[T-1-i][mot]); // la 0-eme est la plus récente
	return ligne;
};

Ligne_t ** __tableau_vers_lignes___api_bitget_v0(char *** tableau, uint T) {
	//0    1    2      3    4    5   6     7      8                 9
	//Unix,Date,Symbol,Open,High,Low,Close,Volume,Volume Base Asset,tradecount
	//float lignes[dar_I][dar_L][dar_N][lignes];
	Ligne_t ** lignes = alloc<Ligne_t*>(5);
	lignes[0] = extraire("prixs"      , tableau, T, 6);
	lignes[1] = extraire("hight"      , tableau, T, 4);
	lignes[2] = extraire("low"        , tableau, T, 5);
	lignes[3] = extraire("volume BTC" , tableau, T, 7);
	lignes[4] = extraire("volume USDT", tableau, T, 8);
	return lignes;
};

//float *** ___eurodollar(char ** tableau, uint lignes);

//	--------------------------------------------------

//	--------------------------------------------------

static uint global_T = 0;

float * ema(float * ligne, float K) {
	float * ret = alloc<float>(global_T);
	ret[0] = ligne[0];
	FOR(1, i, global_T) {
		ret[i] = ret[i-1]*(1.0 - 1.0/K) + ligne[i] * (1.0/K);
	}
	return ret;
};

float * __diff(float * l) {
	float * ret = alloc<float>(global_T);
	ret[0] = 0;
	FOR(1, i, global_T) ret[i] = l[i] - l[i-1];
	return ret;
};

float * __hausse(float * l, float heure) {
	float * ret = alloc<float>(global_T);
	uint _heure = (uint)roundf(heure);
	FOR(0, i, _heure) ret[i] = 0;
	FOR(_heure, i, global_T) ret[i] = l[i] / l[i-_heure] - 1;
	return ret;
};

float * __delta_ema(float * l, float e0, float e1) {
	float * ema_0 = ema(l, e0);
	float * ema_1 = ema(l, e1);
	float * ret = alloc<float>(global_T);
	FOR(0, i, global_T) ret[i] = ema_0[i] - ema_1[i];
	free(ema_1); free(ema_0);
	return ret;
};

void multiplier(float * l, float alpha) {
	FOR(0, i, global_T) l[i] *= alpha;
};

float * __macd(float * l, float K) {
	float * ema12 = ema(l, K*12);
	float * ema26 = ema(l, K*26);
	float * macd = alloc<float>(global_T);
	FOR(0, i, global_T) macd[i] = ema12[i] - ema26[i];
	float * ema9 = ema(macd, K*9);
	float * ret = alloc<float>(global_T);
	FOR(0, i, global_T) ret[i] = macd[i] - ema9[i];
	free(ema12);
	free(ema26);
	free(ema9);
	free(macd);
	return ret;
};

float * __chiffre(float * l, float chiffre) {
	float * ret = alloc<float>(global_T);
	FOR(0, i, global_T) ret[i] = 2*fabs(l[i]/chiffre - roundf(l[i]/chiffre));
	return ret;
};

float * __rsi(float * l, uint n) {
	float * deltas = alloc<float>(global_T);//__diff(l);
	FOR(1, i, global_T) deltas[i] = l[i] - l[i-1];
	//
	/*float * gains  = alloc<float>(global_T);
	float * pertes = alloc<float>(global_T);
	FOR(0, i, global_T) {
		gains [i] = 0;
		pertes[i] = 0;
		if (deltas[i] > 0) gains [i] = deltas[i];
		if (deltas[i] < 0) pertes[i] = -deltas[i];
	}*/
	//
	//float moy_gains  = 0; FOR(0, i, n) moy_gains  += gains [i] / (float)n;
	//float moy_pertes = 0; FOR(0, i, n) moy_pertes += pertes[i] / (float)n;
	//
	float * rsi = alloc<float>(global_T);
	FOR(0, i, n) rsi[i] = 0;
	//
	//rsi[n-1] = (moy_pertes==0 ? 100.0 : (100.0 - (100.0 / (1.0 + (moy_gains/moy_pertes)))));
	//
	FOR(n, i, global_T) {
		//moy_gains  = (moy_gains  - gains [i-n]/(float)n + gains [i]/(float)n);
		//moy_pertes = (moy_pertes - pertes[i-n]/(float)n + pertes[i]/(float)n);
		//
		float mg=0, mp=0;
		FOR(0, j, n) (deltas[i-j]>0?mg:mp) += fabs(deltas[i-j]);
		//
		float rsi_val = 100.0;
		if (mp != 0) {
			rsi_val = 100.0 - 100.0/(1+mg/mp);//(100.0 / (1.0 + moy_gains/moy_pertes));
		}
		rsi[i] = rsi_val;// / 100.0;
	};
	//
	free(deltas);
	//free(gains);
	//free(pertes);
	//
	return rsi;
};

float * __stoch_rsi(float * l, uint n) {
	float * rsi = __rsi(l, n);
	//
	float * ret = alloc<float>(global_T);
	FOR(0, i, global_T) ret[i] = 0;
	//
	FOR(n, i, global_T) {
		float rsi_min=rsi[i], rsi_max=rsi[i];
		FOR(1, j, n) {
			float val = rsi[i-j];
			if (val < rsi_min) rsi_min = val;
			if (val > rsi_max) rsi_max = val;
		};
		//
		if (rsi_max==rsi_min) ret[i] = 0;
		else {
			ret[i] = (rsi[i] - rsi_min) / (rsi_max - rsi_min);
		}
	};
	free(rsi);
	return ret;
};

float * volBU(float * vbtc, float * vusdt, float * prix_btc_en_usdt) {
	float * ret = alloc<float>(global_T);
	FOR(0, i, global_T) {
		float a_moins_b = vbtc[i]*prix_btc_en_usdt[i] - vusdt[i];
		float a_plus__b = (vbtc[i]*prix_btc_en_usdt[i]+vusdt[i])/2;
		ret[i] = a_moins_b/a_plus__b;
	}
	return ret;
};

float * __log(float * f) {
	float * ret = alloc<float>(global_T);
	FOR(0, i, global_T) {
		ret[i] = logf(fabs(f[i]))*(f[i]>=0 ? 1:-1);
	}
	return ret;
};

void __lignes_vers_ema___api_bitget_v0(Ligne_t ** lignes, uint N, char * dar) {
	ASSERT(strcmp(lignes[0]->nom, "prixs"      ) == 0);
	ASSERT(strcmp(lignes[1]->nom, "hight"      ) == 0);
	ASSERT(strcmp(lignes[2]->nom, "low"        ) == 0);
	ASSERT(strcmp(lignes[3]->nom, "volume BTC" ) == 0);
	ASSERT(strcmp(lignes[4]->nom, "volume USDT") == 0);
	//
	uint T = lignes[0]->T;
	global_T = T;
	//
	uint I = 4;	//	Intervs
	uint INTERVS[4] = {1,4,16,64};
	//
	uint L = 13;
	//
#define MAX_INTERV_MULTIPLE 1
	//
	float * lignes_interv[I][L];
	//
	FOR(0, i, I) {
		float * prixs    = ema(lignes[0]->ligne, (float)INTERVS[i]);
		float * hight    = ema(lignes[1]->ligne, (float)INTERVS[i]);
		float * low      = ema(lignes[2]->ligne, (float)INTERVS[i]);
		float * vol_BTC  = ema(lignes[3]->ligne, (float)INTERVS[i]);
		float * vol_USDT = ema(lignes[4]->ligne, (float)INTERVS[i]);
		//
		//
		float heure = (float)INTERVS[i];
		//
		float * prixs1      = __hausse(            prixs         ,heure);  multiplier(prixs1,  25.0);
		float * prixs4      = __hausse(        ema(prixs, 4.0   ),heure);  multiplier(prixs4, 25.0);
		float * delta_26_12 =         (__delta_ema(prixs, 26, 12));  multiplier(delta_26_12, 0.0007);
		float * delta_13_6  =         (__delta_ema(prixs, 13,  6));  multiplier(delta_13_6, 0.0005);
		//
		float * macd1       =         (__macd     (prixs, 1     ));  multiplier(macd1, 0.001);
		float * macd4       =    __log(__macd     (prixs, 4     ));  multiplier(macd4, 0.25);
		//
		float * chiffre1k   =         (__chiffre  (prixs, 1000 ));   multiplier(chiffre1k, 1.0);
		float * chiffre10k  =         (__chiffre  (prixs, 10000));   multiplier(chiffre10k, 1.0);
		//
		float * rsi14       =         (__rsi      (prixs, 14     )); multiplier(rsi14, 0.01);
		float * stoch_rsi14 =         (__stoch_rsi(prixs, 14     )); multiplier(stoch_rsi14, 1.0);
		//
		float * volume_A    = __hausse(        ema(vol_BTC ,10),heure);    multiplier(volume_A, 2.0);
		float * volume_B    = __hausse(        ema(vol_USDT,10),heure);    multiplier(volume_B, 2.0);
		float * volume_AB   = ema(volBU(vol_BTC,vol_USDT,prixs),10); multiplier(volume_AB, 500);
		//
		//
		lignes_interv[i][ 0] = prixs1;
		lignes_interv[i][ 1] = prixs4;
		lignes_interv[i][ 2] = delta_26_12;
		lignes_interv[i][ 3] = delta_13_6;
		lignes_interv[i][ 4] = macd1;
		lignes_interv[i][ 5] = macd4;
		lignes_interv[i][ 6] = chiffre1k;
		lignes_interv[i][ 7] = chiffre10k;
		lignes_interv[i][ 8] = rsi14;
		lignes_interv[i][ 9] = stoch_rsi14;
		lignes_interv[i][10] = volume_A;
		lignes_interv[i][11] = volume_B;
		lignes_interv[i][12] = volume_AB;
	}
	MSG("Lignes écrites !");
	//
	uint DEPART = MAX_INTERV_MULTIPLE * INTERVS[I-1] * N;
	//
	//##########################################################
	//###############  Ecrire le dar.bin  ######################
	//
	FILE * fp = fopen(dar, "wb");
	//
	uint X = I * L * N;
	uint Y = 1 * L * 1;
	//
	uint T_DEPART = T - DEPART;
	FWRITE(&T_DEPART, sizeof(uint), 1, fp);
	FWRITE(&X, sizeof(uint), 1, fp);
	FWRITE(&Y, sizeof(uint), 1, fp);
	FWRITE(&L, sizeof(uint), 1, fp);
	FWRITE(&N, sizeof(uint), 1, fp);
	//
	FWRITE(lignes[0]->ligne+DEPART, sizeof(float), T-DEPART, fp);	//prixs
	//
	//	x__d
	FOR(DEPART, t, T)
		FOR(0, i, I)
			RETRO_FOR(0, n, N)
				FOR(0, l, L)
					FWRITE(&lignes_interv[i][l][t - n*INTERVS[i]], sizeof(float), 1, fp);
	//
	//	y__d
	float zero = 0;
	FOR(DEPART, t, T) {
		FOR(0, l, L) {
			if (t == T-1) {FWRITE(&zero,                                sizeof(float), 1, fp);}
			else          {FWRITE(&lignes_interv[0][l][t+1*INTERVS[0]], sizeof(float), 1, fp);}
		}
	}
	//
	fclose(fp);
	MSG("Dar.bin écrit !");
};

//	--------------------------------------------------

void btcusdt_csv(char * csv, char * dar) {
	//	CSV.csv -> tableau -> Ligne_t* -> Transformations_t* -> dar.bin
	//	-- Lire Fichier ---
	FILE * fp = fopen(csv, "r");
	char * txt = lire_chars(fp);
	fclose(fp);

	uint T = nombre_lignes(txt) - 2;

	char *** tableau = tableau_csv(txt);
	MSG("CSV étrait !");

	//	-- Lire les lignes --
	uint N = 32;
	__lignes_vers_ema___api_bitget_v0(
		__tableau_vers_lignes___api_bitget_v0(tableau, T),
		N,
		dar
	);
}