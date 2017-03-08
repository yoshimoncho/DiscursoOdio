# -*- coding: utf-8 -*-
import codecs
import re

class Limpiador:
    
    #Relacionado a la clase
    lineas = []
    ruta = ""
    rutaout = ""

    ##Relacionado a la limpieza
    emoji_pattern = ""
    re_urls = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    re_hastag = re.compile(r'\#[0-9a-zA-Z]+')
    re_emoji = None
    re_tuser = re.compile(r'@[a-zA-Z0-9_]+')
    re_number = re.compile(r'[0-9]+')
    re_allowed = re.compile(r'[^a-zA-Z\t\s]')
    re_triplep = re.compile(r"[.]+")
    re_limpiaespacios = re.compile(r"[\s\t]+")
    re_limpiarinterr = re.compile(r"\?+")
    re_limpiaprincipioespacios = re.compile(r"^[\s\t]")
    re_aaa = re.compile(r"aa[a]+")
    re_bbb = re.compile(r"bb[b]+")
    re_ccc = re.compile(r"cc[c]+")
    re_ddd = re.compile(r"dd[d]+")
    re_eee = re.compile(r"ee[e]+")
    re_fff = re.compile(r"ff[f]+")
    re_ggg = re.compile(r"gg[g]+")
    re_hhh = re.compile(r"hh[h]+")
    re_iii = re.compile(r"ii[i]+")
    re_jjj = re.compile(r"jj[j]+")
    re_kkk = re.compile(r"kk[k]+")
    re_lll = re.compile(r"ll[l]+")
    re_mmm = re.compile(r"mm[m]+")
    re_nnn = re.compile(r"nn[n]+")
    re_ooo = re.compile(r"oo[o]+")
    re_ppp = re.compile(r"pp[p]+")
    re_qqq = re.compile(r"qq[q]+")
    re_rrr = re.compile(r"rr[r]+")
    re_sss = re.compile(r"ss[s]+")
    re_ttt = re.compile(r"tt[t]+")
    re_uuu = re.compile(r"uu[u]+")
    re_vvv = re.compile(r"vv[v]+")
    re_ww = re.compile(r"ww")
    re_wwww = re.compile(r"www[w]+")
    re_xx = re.compile(r"xx")
    re_xxxx = re.compile(r"xxx[x]+")
    re_yyy = re.compile(r"yy[y]+")
    re_zzz = re.compile(r"zz[z]+")
    re_end_cleaner = re.compile(r"[\s\t]+$")

    def __init__ (self,path):
        self.lineas = []
        file = codecs.open(path, "r", "utf-8")
        self.ruta = path
        for linea in file:
            self.lineas.append(linea)
        print("Tamaño del fichero: "+str(len(self.lineas)))
        file.close()
        cadena = ""
        prefix = "\U0001f"
        for i in range(int("FFF",16)+1):
            hexa = hex(i)[2:]
            if len(hexa) == 1:
                cadena = cadena + prefix + "00"+hexa+"|"
            elif len(hexa) == 2:
                cadena = cadena + prefix + "0"+hexa+"|"
            else:
                cadena = cadena + prefix + hexa+"|"
        self.emoji_pattern = cadena[:-1]
        self.re_emoji = re.compile(self.emoji_pattern.decode('unicode-escape'),re.UNICODE)
            
    def imprime(self):    
        campos = None
        i = 0
        for linea in self.lineas:
            campos = linea.split(";||;")
            if len(campos) != 6:
                print("Error en el numero de campos")
            else:
                texto = campos[4]
                print(str(i)+"-Contenido: " + texto)
                i+=1
        print("_______________________________________________________")
                
    def limpia(self,ruta="emospace2"):
        i = 0

        filename = ruta.split("/")

        #Usaremos solo la parte del documento, quitando directorio por eso el -1
        rutaaux = filename[-1].split(".")
        self.rutaout = "cleancorpus/"+rutaaux[0]+"_clean.txt"
        file = codecs.open(self.rutaout, "w", "utf-8")

        for linea in self.lineas:
            campos = linea.split(";||;")
            if len(campos) != 6:
                print(campos)
                print("Error en el numero de campos")
            else:
                texto = campos[4]
                file.write(str(i)+";||;" + self.clean(texto).encode('unicode-escape')+"\n")
                i+=1
        file.close()
        return self.rutaout

    def clean(self,line):

        line = self.re_emoji.sub(" ",line)
        line = line.lower()

        line = line.replace("\n", ". ").replace("\r", ". ").replace(u"\u0085", ". ").replace(u"\u2028", ". ").replace(u"\u2029", ". ")
        line = self.re_urls.sub(" URL ", line)
        line = self.re_tuser.sub(" USER ", line)
        line = self.re_number.sub(" ", line)
        line = line.replace("."," ").replace("?"," ").replace(";", " ").replace("!"," ")
        line = line.replace(u"¿", " ")
        line = line.replace(u"¡", " ")
        line = line.replace(u"€", " ")
        line = line.replace(u"%", " ")
		 #line = line.replace(".", " DOT ")
		 #line = line.replace(",", " COMMA ")
        line = line.replace(",", " ")
        line = line.replace("/", " ")
        line = line.replace("\\", " ")
        line = line.replace("\"", " ")
        line = line.replace("-", " ")
        line = line.replace(u"&", " ")
        line = line.replace(";", " ")
        line = line.replace(")", " ")
        line = line.replace("(", " ")
        line = line.replace("'", " ")
        line = line.replace(":", " ")
        line = line.replace(u"’", " ")
        line = line.replace(u"‘", " ")
        line = line.replace(u"º", " ")
        line = line.replace("+", " ")
        line = line.replace("_", " ")
        line = line.replace(u"´", " ")
        line = line.replace(u"`", " ")
        line = line.replace(u"", " ")
        line = line.replace(u"", " ")
        line = line.replace(u"", " ")
        line = line.replace(u"", " ")
        line = line.replace(u"🐾", " ")
        line = line.replace(u"🙈", " ")
        line = line.replace(u"⛄", " ")
        line = line.replace(u"💋", " ")
        line = line.replace(u"🎄", " ")
        line = line.replace(u"✌", " ")
        line = line.replace(u"—", " ")
        line = line.replace(u"✈", " ")
        line = line.replace(u"✅", " ")
        line = line.replace(u"😂", " ")
        line = line.replace(u"😜", " ")
        line = line.replace(u"😍", " ")
        line = line.replace(u"️", " ")
        line = line.replace(u"❤", " ")
        line = line.replace(u"«", " ")
        line = line.replace(u"»", " ")
        line = line.replace(u"“", " ")
        line = line.replace(u"”", " ")
        line = line.replace(u"¨", " ")
        line = line.replace(u"á", "a")
        line = line.replace(u"é", "e")
        line = line.replace(u"í", "i")
        line = line.replace(u"ó", "o")
        line = line.replace(u"ú", "u")
        line = line.replace(u"ü", "u")
        line = line.replace(u"ñ", "n")
        line = line.replace(u"…", " ")
        line = line.replace("https", " ")
        line = line.replace("http", " ")
        line = self.re_hastag.sub(" HASHTAG ", line)
        line = self.re_allowed.sub("",line)
        line = self.re_triplep.sub(" ",line)
        line = self.re_limpiaespacios.sub(" ",line)
        line = self.re_limpiarinterr.sub("?",line)
        line = self.re_limpiaprincipioespacios.sub("",line)
        line = self.re_aaa.sub("a",line)
        line = self.re_bbb.sub("b",line)
        line = self.re_ccc.sub("c",line)
        line = self.re_ddd.sub("d",line)
        line = self.re_eee.sub("e",line)
        line = self.re_fff.sub("f",line)
        line = self.re_ggg.sub("g",line)
        line = self.re_hhh.sub("h",line)
        line = self.re_iii.sub("i",line)
        line = self.re_jjj.sub("j",line)
        line = self.re_kkk.sub("k",line)
        line = self.re_lll.sub("l",line)
        line = self.re_mmm.sub("m",line)
        line = self.re_nnn.sub("n",line)
        line = self.re_ooo.sub("o",line)
        line = self.re_ppp.sub("p",line)
        line = self.re_qqq.sub("q",line)
        line = self.re_rrr.sub("r",line)
        line = self.re_sss.sub("s",line)
        line = self.re_ttt.sub("t",line)
        line = self.re_uuu.sub("u",line)
        line = self.re_vvv.sub("v",line)
        line = self.re_xxxx.sub("x",line)
        line = self.re_xx.sub("x",line)
        line = self.re_yyy.sub("y",line)
        line = self.re_zzz.sub("z",line)
        line = self.re_ww.sub("w",line)
        line = self.re_wwww.sub("w",line)
        line = self.re_end_cleaner.sub("",line)

        return line
            
