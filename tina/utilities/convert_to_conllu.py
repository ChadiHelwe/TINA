import stanza
from stanza.utils.conll import CoNLL


import stanza
from stanza.utils.conll import CoNLL


def read_in_chunks(file_object, chunk_size=1024):
    """Lazy function (generator) to read a file piece by piece.
    Default chunk size: 1k."""
    while True:
        data = file_object.readlines(chunk_size)
        if not data:
            break
        yield data


def convert_to_conllu(data, nb_instances=2):
    nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse", use_gpu=True)

    with open(f"{data}", "r") as in_f:
        with open("examples_conllu.txt", "w") as out:
            cnt = 0
            for piece in read_in_chunks(in_f):
                for i in piece:
                    try:
                        doc = nlp(i)  # doc is class Document
                        dicts = (
                            doc.to_dict()
                        )  # dicts is List[List[Dict]], representing each token / word in each sentence in the document
                        conll = CoNLL.convert_dict(dicts)
                        for i in conll:
                            for j in i:
                                if j[3] == "VERB":
                                    j[3] = j[5]
                                out.write("\t".join(j))
                                out.write("\n")
                        out.write("\n")
                        cnt += 1
                        if cnt % 1000 == 0:
                            print(cnt)
                        if cnt >= nb_instances:
                            break
                    except Exception as e:
                        print(e.args)
                if nb_instances is not None and cnt >= nb_instances:
                    break


if __name__ == "__main__":
    convert_to_conllu("examples.txt", None)
