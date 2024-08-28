import struct as st
import matplotlib.pyplot as plt

def normer_11(l):
	_min, _max = min(l), max(l)
	return [2*(e-_min)/(_max-_min)-1 for e in l]

####################################################################################

from calcule import calcule

DEPART = 64*32

T, Y, les_predictions, les_prixs = calcule(
	HEURES =DEPART + 24*5,
	d      ='1H'
)

print(f'len(les_predictions)={len(les_predictions)}, len(les_prixs)={len(les_prixs)}')
assert len(les_predictions) == len(les_prixs)*Y

####################################################################################

flotant = lambda f: '{a:+13f}'.format(a=f)

L = Y

noms_L = [
	"prixs1",
	"prixs4",
	"delta_26_12",
	"delta_13_6",
	"macd1",
	"macd4",
	"chiffre1k",
	"chiffre10k",
	"rsi14",
	"stoch_rsi14",
	"volume_A",
	"volume_B",
	"volume_AB"
]

print("---- Prixs & Prédictions ----")
print("prixs        : ", list(map(flotant, les_prixs      [-8:])))
for l in range(L):
	print(
		"{a:15s}".format(a=noms_L[l]),
		list(map(flotant, [les_predictions[i] for i in range(Y*T) if i%Y==l][-8:]))
	)

les_predictions_prixs = [les_predictions[i] for i in range(Y*T) if i%Y==0]

####################################################################################

prixs_normées = normer_11(les_prixs)

plt.plot(prixs_normées              )
plt.plot(les_predictions_prixs, 'mo')

#	Fleches
for i in range(len(les_prixs)):
	pred       = les_predictions_prixs[i]
	point_prix = prixs_normées  [i]
	if pred >= 0.0: plt.plot([i, i+1], [point_prix, point_prix + 0.03], 'g')
	else:           plt.plot([i, i+1], [point_prix, point_prix - 0.03], 'r')

plt.show()

####################################################################################

fig, ax = plt.subplots(2)

LEVIERS = [10, 25, 50, 125]
__sng = lambda x: (1 if x > 0 else -1)

#
for L in LEVIERS:
	u = 100
	_u0 = []
	#La dernière pred est pour dans 1h (et on sait pas encore le prixs dans 1h)
	for i in range(len(les_predictions_prixs) - 1):
		u += u * L * les_predictions_prixs[i] * (les_prixs[i+1]/les_prixs[i]-1)
		_u0 += [u]
		if u < 0: u = 0
	#
	ax[0].plot(
		_u0,
		label='*'+str(L)
	)
ax[0].set_title("Si le %% de la mise est l'amplitude de la prédiction")
ax[0].legend()
#
for L in LEVIERS:
	u = 100
	_u0 = []
	#La dernière pred est pour dans 1h (et on sait pas encore le prixs dans 1h)
	for i in range(len(les_predictions_prixs) - 1):
		u += u * L * __sng(les_predictions_prixs[i]) * (les_prixs[i+1]/les_prixs[i]-1)
		_u0 += [u]
		if u < 0: u = 0
	#
	ax[1].plot(
		_u0,
		label='sng(p)*'+str(L)
	)
ax[1].set_title("%% mise = signe de la prédiction")
ax[1].legend()
#
plt.show()