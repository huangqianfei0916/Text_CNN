# splite the sequence
# kmer :b = [string[i:i + 3] for i in range(len(string)) if i < len(string) - 2]
# normal :b = [string[i:i + kmer] for i in range(0, len(string), kmer) if i < len(string) - k]
def save_wordfile(fastafile, splite, kmer):
    f = open(fastafile)
    word_file="word.txt"
    f1 = open(word_file, "w")
    k = kmer - 1
    documents = f.readlines()

    string = ""
    flag = 0
    for document in documents:
        if document.startswith(">") and flag == 0:
            flag = 1
            continue
        elif document.startswith(">") and flag == 1:
            if splite == 0:
                b = [string[i:i + kmer] for i in range(len(string)) if i < len(string) - k]
            else:
                b = [string[i:i + kmer] for i in range(0, len(string), kmer) if i < len(string) - k]
            word = " ".join(b)
            f1.write(word)
            f1.write("\n")
            string = ""
        else:
            string += document
            string = string.strip()
    if splite == 0:
        b = [string[i:i + kmer] for i in range(len(string)) if i < len(string) - k]
    else:
        b = [string[i:i + kmer] for i in range(0, len(string), kmer) if i < len(string) - k]
    word = " ".join(b)
    f1.write(word)
    f1.write("\n")
    print("words have been saved in file {}！\n".format(word_file))
    f1.close()
    f.close()


def splite_word(trainfasta_file, kmer,splite):
    train_file = trainfasta_file

    # train set transform to word
    save_wordfile(train_file,splite, kmer)

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parameter of train set
    parser.add_argument('-fasta', required=True,help="fasta file")
    parser.add_argument('-kmer', default=3)
    parser.add_argument('-splite', default=0)
    opt = parser.parse_args()

    print(opt)
    splite_word(opt.fasta,opt.kmer,opt.splite);
