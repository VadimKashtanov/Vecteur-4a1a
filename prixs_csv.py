#! /usr/bin/python3

if __name__ == "__main__":
	from sys import argv
	csv = argv[1]
	with open(csv, 'r') as co:
		text = co.read()
		text = text.split('\n')[2:]
		text = [ligne.split(',') for ligne in text]
		prixs = [float(c) for _,_,_,o,h,l,c,vU,vB,_ in text][::-1]

	import matplotlib.pyplot as plt
	plt.plot(prixs); plt.show()