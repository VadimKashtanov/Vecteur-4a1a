#include "main.cuh"

#include "../impl_template/tmpl_etc.cu"

int main(int argc, char ** argv) {
	//	-- Ecrire Données --
	ASSERT(argc == 3);	//./prog_btcusdt csv.csv dar.bin
	btcusdt_csv(argv[1], argv[2]);
};