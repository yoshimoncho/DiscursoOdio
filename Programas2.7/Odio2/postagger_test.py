from postagger import SpanishParser, traductor


parser = SpanishParser(morphology_path="morfo/es-morphology2.txt",context_path="morfo/es-context2.txt",lexicon_path= "morfo/es-lexicon4.txt")
res = parser.parse("mi nin TOKENWTF",tokenize=True)
resultado = res.split()[0]
print(resultado)