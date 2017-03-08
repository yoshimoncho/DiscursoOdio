
import codecs
import re

f1 = codecs.open("emospace2_clean.txt","r","utf-8")
f2 = codecs.open("emospace_clean.txt","r","utf-8")

ex = re.compile("\b+")
lector1 = f1.readline()
lector2 = f2.readline()


while lector1 != "":
	aux1 = re.split(r"\s+",lector1[:-1])
	aux2 = re.split(r"\s+",lector2[:-1])
	if( aux1 != aux2):
		print("Lector1: " + str(aux1) + "Longitud " + str(len(aux1)))
		print("Lector2: " + str(aux2) + "Longitud " + str(len(aux2)))
	lector1 = f1.readline()
	lector2 = f2.readline()
