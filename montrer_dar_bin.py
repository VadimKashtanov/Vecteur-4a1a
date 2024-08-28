#! /usr/bin/python3

import matplotlib.pyplot as plt
from random import randint
import struct as st

def lire(bins, type, taille):
	octés = st.calcsize(type)*taille
	return st.unpack   (type*taille, bins[:octés]), bins[octés:]

with open("dar.bin", "rb") as co:
	bins = co.read()

	#print(len(bins))

	(T,X,Y,L,N), bins = lire(bins, 'I', 5)
	prixs, bins = lire(bins, 'f', T*1)
	x    , bins = lire(bins, 'f', T*X)
	y    , bins = lire(bins, 'f', T*Y)

	I = int(X/(L*N))

print(f"3 derniers prixs : {prixs[-3:]}")

#	===================================================

plt.plot(prixs); plt.show()

#	===================================================

lignes = [
	[x[t*X + 0*L*N + 0*L + l] for t in range(T)]
	for l in range(L)
]

fig, ax = plt.subplots(L)
for l in range(L):
	ax[l].plot(lignes[l])
plt.show()

#	===================================================

t = randint(0, T-1)
bloque = x[t*X : (t+1)*X]

fig,ax = plt.subplots(1,I)

for i in range(I):
	cadran = bloque[i*N*L : (i+1)*N*L]
	ax[i].imshow([[cadran[n*L + l] for l in range(L)] for n in range(N)])
plt.show()