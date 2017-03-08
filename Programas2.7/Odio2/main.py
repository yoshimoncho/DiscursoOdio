# -*- coding: utf-8 -*-
from limpiador import Limpiador
from postagger import SpanishParser, traductor
import codecs




def main(archivo="rawcorpus/basegrande1.txt"):
    parser = SpanishParser(morphology_path="morfo/es-morphology2.txt",context_path="morfo/es-context2.txt",lexicon_path= "morfo/es-lexicon4.txt")
    l = Limpiador(archivo)
    rutaout = l.limpia(archivo)
    print("-----Limpieza Finalizada-----")
    print("-----Iniciacion De Procesos Optativos-----")
    file = codecs.open(rutaout,"r","utf-8")
    rutaout = rutaout.split("/")[-1]
    fout = codecs.open("treatcorpus/treat_"+rutaout,"w", "utf-8")
    for line in file:
        fragments = line.split(";||;")
        cadena = parser.parse(fragments[1],tokenize=True)
        if(len(cadena) > 1):
            cadenas = cadena.split()[0]
            for term in cadenas:
                meta = traductor(term[1])
                text = term[0]
                if meta in ["Adjetivo","Verbo","Nombre"]:
                    fout.write(text+" ")
        fout.write("\n")
    file.close()
    fout.close()




if __name__ == "__main__":
    main()

