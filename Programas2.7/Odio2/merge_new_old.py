from training_set_generator import extract_labels
import codecs







"""
Esta funcion permite reaprovechar tweets etiquetados anteriormente
para rellenar los nuevos que hayan pasado por el filtro,
asi el etiquetado se hace mas facil permitiendo ver solo los nuevos
"""


def merge_function(new_file,old_file,output_file):
	old_dict , _ = extract_labels(old_file)
	f = codecs.open(new_file,"r", "utf-8")
	fout = codecs.open(output_file,"w", "utf-8")

	for line in f:
		frags = line.split(";||;")
		if len(frags) != 3:
			print("Format of data error!")

		number = int(frags[0][3:])
		if number not in old_dict:
			fout.write(line)

	f.close()
	fout.close()


def used_oldnew2tag(new_file_tagged,old_file_tagged,new_file,output_file):
	old_dict, _ = extract_labels(old_file_tagged)
	recent_dict, _ = extract_labels(new_file_tagged)
	old_dict.update(recent_dict)
	print(len(old_dict))
	f = codecs.open(new_file,"r", "utf-8")
	fout = codecs.open(output_file,"w", "utf-8")

	for line in f:
		frags = line.split(";||;")
		if len(frags) != 3:
			print("Format of data error!")

		line_nojump = line[:-1]
		fout.write(line_nojump + str(old_dict[int(frags[0][3:])])+"\n")

	f.close()
	fout.close()







if __name__ == "__main__":
	#merge_function("tagger_output/new_output.txt","training_set/tagged_1000.txt","tagger_output/merge_output.txt")
	used_oldnew2tag("tagger_output/merge_output2.txt","training_set/tagged_1000.txt","tagger_output/new_output.txt","training_set/tagged_3000.txt")