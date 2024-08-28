from os import system
import struct as st

def lire(bins, taille):
	I = st.calcsize(taille)
	return list(st.unpack(taille, bytes(bins[:I]))), bins[I:]

def calcule(HEURES, d='1H'):
	#system(f"python3 données_bitget.py {d} {HEURES} bitgetBTCUSDT.csv")
	
	system("./prog_btcusdt bitgetBTCUSDT.csv dar_tester_le_model.bin")

	####################################################################################

	system(f"./prog_tester_le_mdl")

	with open("les_predictions.bin", 'rb') as co:
		bins = co.read()
		#
		(T,Y),           bins = lire(bins, 'II')
		les_prixs,       bins = lire(bins, 'f'*T)
		les_predictions, bins = lire(bins, 'f'*T*Y)

	#system("rm les_predictions.bin bitgetBTCUSDT.csv dar_tester_le_model.bin")
	system("rm les_predictions.bin dar_tester_le_model.bin")

	return T, Y, les_predictions, les_prixs