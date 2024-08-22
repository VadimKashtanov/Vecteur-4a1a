#! /usr/bin/python3

import matplotlib.pyplot as plt

import struct as st

def lire(bins, type, taille):
	octés = st.calcsize(type)*taille
	return st.unpack   (type*taille, bins[:octés]), bins[octés:]

with open("prixs/dar.bin", "rb") as co:
	bins = co.read()

	#print(len(bins))

	(T,X,Y,L,N), bins = lire(bins, 'I', 5)
	prixs, bins = lire(bins, 'f', T*1)
	x    , bins = lire(bins, 'f', T*X)
	y    , bins = lire(bins, 'f', T*Y)

print(f"3 derniers prixs : {prixs[-3:]}")

plt.plot(prixs); plt.show()

lignes = [
	[x[t*X + 0*L + l] for t in range(T)]
	for l in range(L)
]

fig, ax = plt.subplots(L)
for l in range(L):
	ax[l].plot(lignes[l])
plt.show()