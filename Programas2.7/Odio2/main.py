# -*- coding: utf-8 -*-
from limpiador import Limpiador
from postagger import SpanishParser, traductor
import codecs
import pickle


"""
Programa que se encarga de limpiar una lista de ficheros
pasados por parametros, se obtendra el archivo limpiado parcialmente
en cleancorpus y el completado en treatcorpus
"""

def main(archivos):
    parser = SpanishParser(morphology_path="morfo/es-morphology2.txt",context_path="morfo/es-context2.txt",lexicon_path= "morfo/es-lexicon4.txt")
    ref_dict = {}
    for j in range(len(archivos)):
        print("Archivo: "+ archivos[j])
        l = Limpiador(archivos[j])
        rutaout = l.limpia(diccionario=ref_dict,num_doc=j+1,ruta=archivos[j])
        print("-----Limpieza Finalizada-----")
        print("-----Iniciacion De Procesos Optativos-----")
        file = codecs.open(rutaout,"r","utf-8")
        rutaout = rutaout.split("/")[-1]
        fout = codecs.open("treatcorpus/treat_"+rutaout,"w", "utf-8")
        for line in file:
            fragments = line.split(";||;")
            cadena = parser.parse(fragments[1],tokenize=True)
            if(len(cadena) > 1):
                fout.write(fragments[0]+";||;")
                cadenas = cadena.split()[0]
                for term in cadenas:
                    meta = traductor(term[1])
                    text = term[0]
                    if meta in ["Adjetivo","Verbo","Nombre"]:
                        fout.write(text+" ")
            fout.write("\n")
        file.close()
        fout.close()
    pickle.dump(ref_dict,open( "id2docid_posrel.p", "wb" ))





if __name__ == "__main__":
    archivos = ["rawcorpus/basegrande"+str(i)+".txt" for i in range(1,11)]
    main(archivos)

