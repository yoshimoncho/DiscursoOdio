


import codecs

file = codecs.open("basegrande15_3000_2.txt", "r", "utf-8")
aciertos = 0.0
totales = 0.0
acierto_diciendo0 = 0
fallo_diciendo0 = 0
acierto_diciendo1 = 0
fallo_diciendo1 = 0
reales_0 = 0
reales_1 = 0
totales_noodio = 0
totales_odio = 0
for line in file:
	frags = line.split(";||;")
	if len(frags) < 3:
		print("Error de etiquetado")
		exit(-1)
	pred = int(frags[1])

	real = int(frags[2])

	if real == 0:
		if pred == 0:
			acierto_diciendo0 +=1
		else:
			fallo_diciendo0 +=1
	else:
		if pred == 1:
			acierto_diciendo1 +=1
		else:
			fallo_diciendo1 +=1

	if real == 0:
		totales_noodio += 1
	else:
		totales_odio +=1







	if pred == real:
		aciertos +=1.0
	totales += 1.0
file.close()

print("Totales: " + str(totales_noodio + totales_odio))
print("Totales de no odio: " + str(totales_noodio))
print("Totales de odio: " + str(totales_odio))
print("Acierto: " + str(aciertos/totales))
print("(ACIERTO)- Era no odio y predijo no odio: " +str(acierto_diciendo0))
print("(ERROR) - Era no odio y predijo odio: " +str(fallo_diciendo0))
print("(ACIERTO) - Era odio y predijo odio: " +str(acierto_diciendo1))
print("(ERROR) - Era odio y dijo no odio:  " +str(fallo_diciendo1))
