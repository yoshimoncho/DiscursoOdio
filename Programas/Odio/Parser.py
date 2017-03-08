# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 18:13:09 2017

@author: JuanCarlos
"""

from glob import glob
from codecs import open, BOM_UTF8
from collections import defaultdict
from nltk.tag import UnigramTagger
from nltk.tag.brill_trainer import BrillTaggerTrainer
#from nltk.tag.brill import SymmetricProximateTokensTemplate
#from nltk.tag.brill import ProximateTokensTemplate
#from nltk.tag.brill import ProximateTagsRule
#from nltk.tag.brill import ProximateWordsRule
from nltk.tbl.template import Template
from nltk.tag.brill import Pos, Word,fntbl37,brill24
from nltk.tag import tnt
def wikicorpus(words=1000000, start=0):
    s = [[]]
    i = 0
    for f in glob("D:/Proyectos/tagged.es/*")[start:]:
        for line in open(f, encoding="latin-1"):
            if line == "\n" or line.startswith((
              "<doc", "</doc>", "ENDOFARTICLE", "REDIRECT",
              "Acontecimientos", 
              "Fallecimientos", 
              "Nacimientos")):
                continue
            w, lemma, tag, x = line.split(" ")
            if tag.startswith("Fp"):
                tag = tag[:3]
            elif tag.startswith("V"):  # VMIP3P0 => VMI
                tag = tag[:3]
            elif tag.startswith("NC"): # NCMS000 => NCS
                tag = tag[:2] + tag[3]
            else:
                tag = tag[:2]
            for w in w.split("_"): # Puerto_Rico
                s[-1].append((w, tag));i+=1
            if tag == "Fp" and w == ".":
                s.append([])
            if i >= words:
                return s[:-1]

def lexicon_function():
    # "el" => {"DA": 3741, "NP": 243, "CS": 13, "RG": 7}) 
    # defaultdict permite inicializar los valores del diccionario con diccionarios
    # que contengas 0 como valor predeterminado
    lexicon = defaultdict(lambda: defaultdict(int))
      
    # aumento de frecuencias por tag dada una palabra
    for sentence in wikicorpus(1000000):
        for w, tag in sentence:
            lexicon[w][tag] += 1
     
    top = []  
    for w, tags in lexicon.items():    
        freq = sum(tags.values())      # 3741 + 243 + ... (frecuencia absolutas)
        tag  = max(tags, key=tags.get) # DA
        top.append((freq, w, tag))
     
    top = sorted(top, reverse=True)[:100000] # top 100,000
    top = ["%s %s" % (w, tag) for freq, w, tag in top if w]
     
    open("es-lexicon.txt", "w").write(BOM_UTF8 + "\n".join(top).encode("utf-8"))
    
def rules_function():
    sentences = wikicorpus(words=1000000)
    
    ANONYMOUS = "anonymous"
    for s in sentences:
        for i, (w, tag) in enumerate(s):
            if tag == "NP": # NP = proper noun in Parole tagset.
                s[i] = (ANONYMOUS, "NP")
    """
    ctx = [ # Context = surrounding words and tags.
        SymmetricProximateTokensTemplate(ProximateTagsRule,  (1, 1)),
        SymmetricProximateTokensTemplate(ProximateTagsRule,  (1, 2)),
        SymmetricProximateTokensTemplate(ProximateTagsRule,  (1, 3)),
        SymmetricProximateTokensTemplate(ProximateTagsRule,  (2, 2)),
        SymmetricProximateTokensTemplate(ProximateWordsRule, (0, 0)),
        SymmetricProximateTokensTemplate(ProximateWordsRule, (1, 1)),
        SymmetricProximateTokensTemplate(ProximateWordsRule, (1, 2)),
        ProximateTokensTemplate(ProximateTagsRule, (-1, -1), (1, 1)),
    ]
    """
    ctx = [
          Template(Pos((1, 1))), 
          Template(Pos((1, 2))),
          Template(Pos((1, 3))),
          Template(Pos((2,2))),
          Template(Word((0,0))),
          Template(Word((1,1))),
          Template(Word((1,1))),
          Template(Pos((-1, -1)),Pos((1, 1)))
        
    ]
     
    tagger = UnigramTagger(sentences)
    tagger = BrillTaggerTrainer(tagger, ctx , trace=0)
    tagger = tagger.train(sentences, max_rules=100)
    #print (tagger.evaluate(wikicorpus(10000, start=1)))

    for rule in tagger.rules():
        a = rule.original_tag
        b = rule.replacement_tag
        c = rule._conditions
        x = c[0][1]
        r = c[0][0]
        if len(c) != 1:
            continue
        if isinstance(r, Pos):
            if r == Pos([-1]): cmd = "PREVTAG"
            elif r == Pos([1]): cmd = "NEXTTAG"
            elif r == Pos([-2, -1]): cmd = "PREV1OR2TAG"
            elif r == Pos([1, 2]): cmd = "NEXT1OR2TAG"
            elif r == Pos([-3, -1]): cmd = "PREV1OR2OR3TAG"
            elif r == Pos([1, 3]): cmd = "NEXT1OR2OR3TAG"
            elif r == Pos([-2]): cmd = "PREV2TAG"
            elif r == Pos([2]): cmd = "NEXT2TAG"
        if isinstance(r, Word):
            if r == Word([0]): cmd = "CURWD"
            elif r == Word([-1]): cmd = "PREVWD"
            elif r == Word([1]): cmd = "NEXTWD"
            elif r == Word([-2, -1]): cmd = "PREV1OR2WD"
            elif r == Word([1, 2]): cmd = "NEXT1OR2WD"
        print("%s %s %s %s" % (a, b, cmd, x))
        ctx.append("%s %s %s %s" % (a, b, cmd, x))

#print tagger.evaluate(wikicorpus(10000, start=1)) 

def main():
    pass
    



    #print(cadena[0])


if __name__ == "__main__":
    main()