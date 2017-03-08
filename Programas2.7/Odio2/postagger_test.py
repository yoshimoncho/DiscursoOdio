from postagger import SpanishParser, traductor


parser = SpanishParser(morphology_path="es-morphology2.txt",context_path="es-context2.txt",lexicon_path= "es-lexicon4.txt")
res = parser.parse("mi nin URL",tokenize=True)
resultado = res.split(" ")
print(resultado)