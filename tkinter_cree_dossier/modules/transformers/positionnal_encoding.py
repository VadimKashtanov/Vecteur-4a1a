#! /usr/bin/python3

from tkinter_cree_dossier.modules._etc import *

class EMBEDE_POSITIONNAL(Module_Mdl):
	bg, fg = 'light blue', 'black'
	nom = "[Embede+Positionnal]"
	X, Y = [0], [0]
	X_noms, Y_noms = ["X"], ["Y"]
	params = {
		'mot'     : 1,
		'd_model' : 1,
		'mots'    : 1,
	}
	def cree_ix(self):
		#	Params
		X = self.X[0]
		Y = self.Y[0]

		mot     = self.params['mot'    ]
		d_model = self.params['d_model']
		mots    = self.params['mots'   ]

		mot  = int(X / mots)

		#	------------------

		self.elements = {
			'x' : MODULE_i_Y(X=[mot*mots], Y=[mot*mots], params={}).cree_ix(),
			'embede'       : MODULE_i_MatMul_Poid_AP(X=[X], Y=[d_model*mots], params={'Ax':mot, 'Ay':mots, 'Bx':d_model, 'C0':1}).cree_ix(),
			'positionnal'  : MODULE_i_Positionnal   (X=[d_model*mots], Y=[d_model*mots], params={'L':d_model, 'N':mots}).cree_ix(),
		}

		self.connections = {
			'x' : {0:None},

			'embede' : {0:('x',0)},
			'positionnal' : {0:('embede',0)}
		}

		self.cree_elements_connections()
		return self.ix

class ANTI_EMBEDE(Module_Mdl):
	bg, fg = 'light blue', 'black'
	nom = "[ANTI EMBEDE]"
	X, Y = [0], [0]
	X_noms, Y_noms = ["X"], ["Y"]
	params = {
		'dimention' : 1,
		'mots'      : 1,
		'têtes'     : 1,
	}
	def cree_ix(self):
		#	Params
		X = self.X[0]
		Y = self.Y[0]

		dimention = self.params['dimention']
		mots      = self.params['mots'     ]
		têtes     = self.params['têtes'    ]

		Ax        = int(X / mots)

		#	------------------

		self.elements = {
			'x'      : MODULE_i_Y(X=[Ax*mots], Y=[Ax*mots], params={}).cree_ix(),
			'embede' : MODULE_i_MatMul_Poid_AP(X=[X], Y=[dimention*mots], params={'Ax':Ax, 'Ay':mots, 'Bx':dimention, 'C0':1}).cree_ix(),
		}

		self.connections = {
			'x' : {0:None},

			'embede' : {0:('x',0)},
		}

		self.cree_elements_connections()
		return self.ix