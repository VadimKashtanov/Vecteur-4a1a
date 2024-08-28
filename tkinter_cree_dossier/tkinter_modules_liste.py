from tkinter_cree_dossier.modules._etc import *

from tkinter_cree_dossier.modules.tous_les_modules import *

class NONE(Module_Mdl):
	nom = ""
	X,      Y      = [], []
	X_noms, Y_noms = [], []
	params = {}
	def cree_ix(self):
		return []

#	=========================================================================	#

modules = [
	DOT1D,
	DOT1D__CHAINE,
	DOT1D_2X,
	DOT1D_3X,
	DOT2D_AP,
	DOT2D_PA,
	NONE,
	NONE,
	NONE,
#	-----------------	#
#	DOT1D_RECCURENT,
#	DOT1D_RECCURENT__CHAINE,
#	NONE,
#	DOT1D_RECCURENT_N,
#	DOT1D_RECCURENT_N__CHAINE,
#	NONE,
#	NONE,
#	NONE,
#	NONE,
#	-----------------	#
#	LSTM1D_2X,
#	LSTM1D_3X,
#	LSTM1D_PROFOND,
#	NONE,
#	LSTM1D_PEEPHOLE,
#	LSTM1D_PEEPHOLE__CHAINE,
#	NONE,
#	SIGNALE_LOGISTIQUE,
#	NONE,
#	----------------	#
	NONE,
	ENCODEUR,
	ENCODEUR_CHAINE,
	NONE,
	NONE,
	NONE,
	NONE,
	EMBEDE_POSITIONNAL,
	ANTI_EMBEDE,
#	----------------	#
	SOMME3,
	NONE,
	NORMALISATION,
	NONE,
	BATCH_NORM,
	NONE,
	NONE,
	NONE,
	Self_MultiHeadAttention,
#	----------------	#
	STABLE_SOFTMAX,
	NONE,
	NONE,
	NONE,
	NONE,
	UNION_3,
	UNION_4,
	NONE,
	NONE,#SPLIT_4,
#	----------------	#
	MODELE_TRANSFORMER_4,
	MODELE_TRANSFORMER_4_3_2_1,
	NONE,
	NONE,
	NONE,
	NONE,
	NONE,
	NONE,
	NONE,
]

"""
	NONE,
	NONE,
	NONE,
	NONE,
	NONE,
	NONE,
	NONE,
	NONE,
	NONE,
"""