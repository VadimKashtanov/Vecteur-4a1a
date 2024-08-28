from tkinter_cree_dossier.modules._etc import *

from tkinter_cree_dossier.modules.dot.dot1d import *

#	==================================================================

class MultiHeadAttention(Module_Mdl):
	nom = "Scaled Dot Attention : QKV"
	X, Y = [0,0,0], [0]
	X_noms, Y_noms = ["Q", "K", "V"], ["Img"]
	params = {
		'd_model' : 1,
		'mots'    : 1,
		'clef'    : 1,
		'têtes'   : 1,
	}
	def cree_ix(self):
		#	Params
		Q,K,V = self.X
		Y,    = self.Y

		d_model = self.params['d_model']
		mots    = self.params['mots']
		clef    = self.params['clef' ]
		têtes   = self.params['têtes']

		assert Q == K == V == d_model*mots

		QKV = d_model*mots

		self.elements = {
			'Q' : MODULE_i_Canalisation(X=[QKV], Y=[QKV*têtes], params={'C0':têtes}).cree_ix(),
			'K' : MODULE_i_Canalisation(X=[QKV], Y=[QKV*têtes], params={'C0':têtes}).cree_ix(),
			'V' : MODULE_i_Canalisation(X=[QKV], Y=[QKV*têtes], params={'C0':têtes}).cree_ix(),

			'q' : MODULE_i_MatMul_Poid_AP(X=[QKV*têtes], Y=[clef*mots*têtes], params={'Ax':d_model, 'Ay':mots, 'Bx':clef, 'C0':têtes}).cree_ix(),
			'k' : MODULE_i_MatMul_Poid_AP(X=[QKV*têtes], Y=[clef*mots*têtes], params={'Ax':d_model, 'Ay':mots, 'Bx':clef, 'C0':têtes}).cree_ix(),
			'v' : MODULE_i_MatMul_Poid_AP(X=[QKV*têtes], Y=[clef*mots*têtes], params={'Ax':d_model, 'Ay':mots, 'Bx':clef, 'C0':têtes}).cree_ix(),

			'q@k.T' :            MODULE_i_QKtDivClef(X=[clef*mots*têtes, clef*mots*têtes], Y=[mots *mots*têtes], params={'Ax':clef, 'Ay':mots, 'Bx':mots, 'C0':têtes}).cree_ix(),
			'softmax(q@k.T)' :   STABLE_SOFTMAX     (X=[mots*mots*têtes                 ], Y=[mots *mots*têtes], params={'Vect':mots},                               ).cree_ix(),
			'softmax(q@k.T)@v' : MODULE_i_MatMul    (X=[mots*mots*têtes, clef*mots*têtes], Y=[clef*mots*têtes ], params={'Ax':mots, 'Ay':mots, 'Bx':clef, 'C0':têtes}).cree_ix(),

			'concatenation' : MODULE_i_Concatenation(X=[clef*mots*têtes], Y=[(clef*têtes)*mots], params={'Ax':clef, 'Ay':mots, 'C0':têtes}).cree_ix(),
			
			'lineair' : MODULE_i_MatMul_Poid_AP(X=[(clef*têtes)*mots], Y=[(d_model)*mots], params={'Ax':clef*têtes, 'Ay':mots, 'Bx':d_model, 'C0':1}).cree_ix(),
		}

		self.connections = {
			'Q' : {0:None},
			'K' : {0:None},
			'V' : {0:None},
			#
			'q' : {0:('Q',0)},
			'k' : {0:('K',0)},
			'v' : {0:('V',0)},
			#
			#'k.T' : {0:('k',0)},
			#
			'q@k.T'            : {0:('q',0), 1:('k',0)},
			'softmax(q@k.T)'   : {0:('q@k.T', 0)},
			'softmax(q@k.T)@v' : {0:('softmax(q@k.T)', 0), 1 : ('v', 0)},
			#
			'concatenation' : {0:('softmax(q@k.T)@v', 0)},
			#
			'lineair' : {0:('concatenation',0)}
		}

		self.cree_elements_connections()
		return self.ix

class MultiHeadAttention_Sparce(Module_Mdl):
	#
	pass

class Self_MultiHeadAttention(Module_Mdl):
	nom = "Self Scaled Dot Attention : QKV"
	X, Y = [0], [0]
	X_noms, Y_noms = ["X"], ["Y"]
	params = {
		'd_model' : 1,
		'mots'    : 1,
		'clef'    : 1,
		'têtes'   : 1,
	}
	def cree_ix(self):
		#	Params
		X, = self.X
		Y, = self.Y

		d_model = self.params['d_model']
		mots    = self.params['mots']
		clef    = self.params['clef' ]
		têtes   = self.params['têtes']

		self.elements = {
			'x' : MODULE_i_Y(X=[X], Y=[X], params={}).cree_ix(),
			'multihead' : MultiHeadAttention(X=[X,X,X], Y=[Y], params={
				'd_model':d_model,
				'mots':mots,
				'clef':clef,
				'têtes':têtes
				}).cree_ix(),
		}
		self.connections = {
			'x' : {0:None},
			'multihead' : {0:('x',0), 1:('x',0), 2:('x',0)}
		}

		self.cree_elements_connections()
		return self.ix