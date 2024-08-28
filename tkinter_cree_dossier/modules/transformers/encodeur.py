from tkinter_cree_dossier.modules._etc import *

class ENCODEUR(Module_Mdl):
	bg, fg = 'light blue', 'black'
	nom = "[ENCODEUR]"
	X, Y = [0], [0]
	X_noms, Y_noms = ["X"], ["Y"]
	params = {
		'd_model' : 1,
		'mots'    : 1,
		'têtes'   : 1,
		'clef'    : 1,
		'ff'      : 1,
	}
	def cree_ix(self):
		#	Params
		X, = self.X
		Y, = self.Y

		d_model = self.params['d_model']
		mots = self.params['mots']
		têtes = self.params['têtes']
		clef = self.params['clef']
		ff = self.params['ff']

		#	------------------

		self.elements = {
			'x' : MODULE_i_Y(X=[d_model*mots], Y=[d_model*mots], params={}).cree_ix(),
			#
			'MultiHeadAttention'      : Self_MultiHeadAttention(X=[d_model*mots], Y=[d_model*mots], params={'d_model' : d_model,'mots' : mots, 'clef' : clef, 'têtes' : têtes}).cree_ix(),
			'MultiHeadAttention_drop' : MODULE_i_Drop_Vecteur  (X=[d_model*mots], Y=[d_model*mots], params={'VECT':d_model, 'drop %':20}).cree_ix(),
			#
			'Somme 0' : MODULE_i_Somme(X=[d_model*mots,d_model*mots], Y=[d_model*mots], params={}).cree_ix(),
			'Norme 0' : BATCH_NORM    (X=[d_model*mots], Y=[d_model*mots], params={'C0':mots}).cree_ix(),
			#
			'FFN' :      FFN                  (X=[d_model*mots], Y=[d_model*mots], params={'d_model':d_model, 'mots':mots, 'ff':ff}).cree_ix(),
			'FFN_drop' : MODULE_i_Drop_Vecteur(X=[d_model*mots], Y=[d_model*mots], params={'VECT':d_model, 'drop %':20}).cree_ix(),
			#
			'Somme 1' : MODULE_i_Somme(X=[d_model*mots,d_model*mots], Y=[d_model*mots], params={}).cree_ix(),
			'Norme 1' : BATCH_NORM    (X=[d_model*mots], Y=[d_model*mots], params={'C0':mots}).cree_ix(),
		}

		self.connections = {
			'x' : {0:None},
			#
			'MultiHeadAttention' : {0:('x',0)},
			'MultiHeadAttention_drop' : {0:('MultiHeadAttention',0)},
			#
			'Somme 0' : {0:('MultiHeadAttention_drop',0), 1:('x',0)},
			'Norme 0' : {0:('Somme 0',0)},
			#
			'FFN' : {0:('Norme 0',0)},
			'FFN_drop' : {0:('FFN',0)},
			#
			'Somme 1' : {0:('FFN_drop',0), 1:('Norme 0',0)},
			'Norme 1' : {0:('Somme 1',0)},
		}

		self.cree_elements_connections()
		return self.ix

class ENCODEUR_CHAINE(Module_Mdl):
	img = img_chaine
	bg, fg = 'light blue', 'black'
	nom = "[ENCODEUR] Chaine"
	X, Y = [0], [0]
	X_noms, Y_noms = ["X"], ["Y"]
	params = {
		'd_model' : 1,
		'mots'    : 1,
		'têtes'   : 1,
		'clef'    : 1,
		'ff'      : 1,
		#
		'N' : 1,
	}
	def cree_ix(self):
		#	Params
		X, = self.X
		Y, = self.Y

		d_model = self.params['d_model']
		mots = self.params['mots']
		têtes = self.params['têtes']
		clef = self.params['clef']
		ff = self.params['ff']
		N = self.params['N']

		self.elements = {**{
			'-1' : MODULE_i_Y(X=[d_model*mots], Y=[d_model*mots], params={}).cree_ix(),
		}, **{
			f'{i}' : ENCODEUR(X=[d_model*mots],Y=[d_model*mots], params={
				'd_model' : d_model,
				'mots'    : mots,
				'têtes'   : têtes,
				'clef'    : clef,
				'ff'      : ff,
				}).cree_ix()
			for i in range(N)
		}}
		self.connections = {**{
			'-1' : {0:None},
		}, **{
			f'{i}' : {0:(f'{i-1}',0)}
			for i in range(N)
		}}
		self.cree_elements_connections()
		return self.ix