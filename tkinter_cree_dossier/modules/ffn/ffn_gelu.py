from tkinter_cree_dossier.modules._etc import *

class FFN(Module_Mdl):
	nom = "FFN : Gelu(x@P+b)@P+b"
	X, Y = [0], [0]
	X_noms, Y_noms = ["X"], ["Y"]
	params = {
		'd_model' : 1,
		'mots'    : 1,
		'ff'      : 1,
	}
	def cree_ix(self):
		#	Params
		X, = self.X
		Y, = self.Y

		d_model = self.params['d_model']
		mots    = self.params['mots']
		ff      = self.params['ff']

		self.elements = {
			'x' : MODULE_i_Y(X=[d_model*mots], Y=[d_model*mots], params={}).cree_ix(),
			#
			'x@P' : MODULE_i_MatMul_Poid_AP(X=[d_model*mots], Y=[ff*mots], params={'Ax':d_model, 'Ay':mots, 'Bx':ff, 'C0':1}).cree_ix(),
			'Gelu(x@P+b)' : MODULE_i_Activation_Poid(X=[ff*mots], Y=[ff*mots], params={'activ':f_GELU}).cree_ix(),
			'Gelu(x@P+b)@P' : MODULE_i_MatMul_Poid_AP(X=[ff*mots], Y=[d_model*mots], params={'Ax':ff, 'Ay':mots, 'Bx':d_model, 'C0':1}).cree_ix(),
			'Gelu(x@P+b)@P+b' : MODULE_i_Activation_Poid(X=[d_model*mots], Y=[d_model*mots], params={'activ':f_GELU}).cree_ix(),
		}

		self.connections = {
			'x' : {0:None},
			#
			'x@P' : {0:('x',0)},
			'Gelu(x@P+b)' : {0:('x@P',0)},
			'Gelu(x@P+b)@P' : {0:('Gelu(x@P+b)',0)},
			'Gelu(x@P+b)@P+b' : {0:('Gelu(x@P+b)@P',0)},
		}

		self.cree_elements_connections()
		return self.ix
