import collections
import os

from load_utils import *


def pre_data(dir_path, out_dir):
    vocab = load_json(dir_path + "vocab.json")
    save_list = [x[0] for x in vocab[:int(0.1 * len(vocab))]]
    tgt_counter = collections.Counter()
    for file in os.listdir(dir_path):
        if "txt" in file:
            if "tgt" in file:
                tgt = load_txt(dir_path + file)
                for line in tgt:
                    tgt_counter.update(line.split())
    tgt_vocab = sorted(tgt_counter.items(), key=lambda x: x[1], reverse=True)
    print(len(save_list))
    save_list.extend(tgt_vocab[:int(0.15 * len(tgt_vocab))])
    save_set = set(save_list)
    print(len(save_set))

    for file in os.listdir(dir_path):
        if "txt" in file:
            if "src" or "tgt" in file:
                data = load_txt(dir_path + file)
                new_data = []
                for line in data:
                    new_line = []
                    old_list = line.split()
                    for word in old_list:
                        if word in save_set:
                            new_line.append("1")
                        else:
                            new_line.append("0")
                    one = " ".join(new_line)
                    assert len(new_line) == len(old_list)
                    new_data.append(one)
                new_name = file.replace("src", "pos_en")
                new_name = new_name.replace("tgt", "pos_de")
                save_txt("\n".join(new_data), out_dir + new_name)


def main():
    pre_data("/home/wangyida/git/data/onmt_opensub/",
             "/home/wangyida/git/data/onmt_opensub/freq/")
    print(1)


if __name__ == '__main__':
    main()
