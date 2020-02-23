import collections
import os
import random

import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

from utils import *

random.seed(2019)


# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')


def onmt_Daily(path, outdir, train=False):
    data = load_txt(path)
    utterances = []

    for line in tqdm(data, mininterval=1):
        dialog = line.strip().split("__eou__")

        for i in range(len(dialog) - 2):
            post_list = nltk.word_tokenize(dialog[i].strip())
            one_post = " ".join(nltk.word_tokenize(dialog[i].strip()))
            resp_list = nltk.word_tokenize(dialog[i + 1].strip())
            one_resp = " ".join(resp_list)

            post_pos = nltk.pos_tag(post_list)
            one_en_pos = " ".join([x[1] for x in post_pos])
            resp_pos = nltk.pos_tag(resp_list)
            one_de_pos = " ".join([x[1] for x in resp_pos])
            assert len(one_post) > 0
            assert len(one_resp) > 0
            assert len(one_de_pos) > 0
            assert len(one_en_pos) > 0
            utterances.append([one_post, one_resp, one_en_pos, one_de_pos])

    if train:
        random.shuffle(utterances)
    post = [x[0] for x in utterances]
    resp = [x[1] for x in utterances]
    pos_en = [x[2] for x in utterances]
    pos_de = [x[3] for x in utterances]

    set_type = path[path.rindex("_") + 1: path.rindex(".")]
    assert len(post) == len(resp) == len(pos_de) == len(pos_en)
    save_txt("\n".join(post), outdir + "src-" + set_type + ".txt")
    save_txt("\n".join(resp), outdir + "tgt-" + set_type + ".txt")
    save_txt("\n".join(pos_en), outdir + "pos_en-" + set_type + ".txt")
    save_txt("\n".join(pos_de), outdir + "pos_de-" + set_type + ".txt")
    print(len(post))


def onmt_reddit(path, outdir):
    data = load_txt(path)
    post = []
    resp = []
    pos_en = []
    pos_de = []
    vocab = collections.Counter()

    random.shuffle(data)
    for line in tqdm(data, mininterval=1):
        dialog = line.strip().split("\t")

        post_list = nltk.word_tokenize(dialog[0].strip())
        one_post = " ".join(post_list)
        resp_list = nltk.word_tokenize(dialog[1].strip())
        one_resp = " ".join(resp_list)

        vocab.update(post_list)
        vocab.update(resp_list)

        post_pos = nltk.pos_tag(post_list)
        one_en_pos = " ".join([x[1] for x in post_pos])
        resp_pos = nltk.pos_tag(resp_list)
        one_de_pos = " ".join([x[1] for x in resp_pos])
        assert len(one_post) > 0
        assert len(one_resp) > 0
        assert len(one_de_pos) > 0
        assert len(one_en_pos) > 0
        post.append(one_post)
        resp.append(one_resp)
        pos_en.append(one_en_pos)
        pos_de.append(one_de_pos)

    assert len(post) == len(resp) == len(pos_de) == len(pos_en)

    vocab_json = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    save_json(vocab_json, "/home/wangyida/data/reddit/200W_reddit_vocab.json")

    train_src, trian_tgt = post[:-20000], resp[:-20000]
    valid_src, valid_tgt = post[-20000:-10000], resp[-20000:-10000]
    test_src, test_tgt = post[-10000:], resp[-10000:]

    pos_train_src, pos_trian_tgt = pos_en[:-20000], pos_de[:-20000]
    pos_valid_src, pos_valid_tgt = pos_en[-20000:-10000], pos_de[-20000:-10000]
    pos_test_src, pos_test_tgt = pos_en[-10000:], pos_de[-10000:]

    try:
        for (src, tgt) in zip([train_src, valid_src, test_src], [trian_tgt, valid_tgt, test_tgt]):
            for set_type in ["train", "valid", "test"]:
                save_txt("\n".join(src), outdir + "src-" + set_type + ".txt")
                save_txt("\n".join(tgt), outdir + "tgt-" + set_type + ".txt")

        for (pos_en, pos_de) in zip([pos_train_src, pos_valid_src, pos_test_src],
                                    [pos_trian_tgt, pos_valid_tgt, pos_test_tgt]):
            for set_type in ["train", "valid", "test"]:
                save_txt("\n".join(pos_en), outdir + "pos_en-" + set_type + ".txt")
                save_txt("\n".join(pos_de), outdir + "pos_de-" + set_type + ".txt")
    except:
        import pdb
        pdb.set_trace()


def convert_vocab():
    vocab_stoi = load_json("./tool_data/stoi.json")
    vocab = [x for x in vocab_stoi.keys()]
    print(vocab[:10])
    vocab_pos_dict = nltk.pos_tag(vocab)
    vocab_pos_dict = {x[0]: x[1] for x in vocab_pos_dict}

    pos_stoi = load_json("./tool_data/pos_stoi.json")
    vocab_pos = {}
    for k, v in vocab_pos_dict.items():
        try:
            vocab_pos[vocab_stoi[k]] = pos_stoi[v]
        except:
            print(1)
    save_json(vocab_pos, "./vocab_pos_dict.json")


def onmt_opensubtitle(dir_path, out_dir):
    file_list = os.listdir(dir_path)
    vocab = collections.Counter()
    for file in file_list:
        if "response" in file:
            continue
        name = file[file.rindex("_") + 1: file.rindex(".")]
        print(name)
        post_path = dir_path + file
        post = load_txt(post_path)
        response = load_txt(post_path.replace("post", "response"))
        data = [[x, y] for x, y in zip(post, response)]
        if "train" in file:
            random.shuffle(data)

        src = []
        tgt = []
        pos_src = []
        pos_tgt = []

        for dialog in tqdm(data, mininterval=1):
            post_list = nltk.word_tokenize(dialog[0].strip())
            one_post = " ".join(post_list)
            resp_list = nltk.word_tokenize(dialog[1].strip())
            one_resp = " ".join(resp_list)

            vocab.update(post_list)
            vocab.update(resp_list)

            post_pos = nltk.pos_tag(post_list)
            one_en_pos = " ".join([x[1] for x in post_pos])
            resp_pos = nltk.pos_tag(resp_list)
            one_de_pos = " ".join([x[1] for x in resp_pos])
            assert len(one_post) > 0
            assert len(one_resp) > 0
            assert len(one_de_pos) > 0
            assert len(one_en_pos) > 0
            src.append(one_post)
            tgt.append(one_resp)
            pos_src.append(one_en_pos)
            pos_tgt.append(one_de_pos)

        assert len(src) == len(tgt) == len(pos_src) == len(pos_tgt)
        save_txt("\n".join(src), out_dir + "src-" + name + ".txt")
        save_txt("\n".join(tgt), out_dir + "tgt-" + name + ".txt")
        save_txt("\n".join(pos_src), out_dir + "pos_en-" + name + ".txt")
        save_txt("\n".join(pos_tgt), out_dir + "pos_de-" + name + ".txt")
    vocab_json = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    save_json(vocab_json, out_dir + "vocab.json")


def freq_opensub(rootdir, out_dir):
    dir_path = rootdir + "data/"
    file_list = os.listdir(dir_path)
    stopwords = set(load_txt(rootdir + "stopwords.txt"))
    print(len(stopwords))
    vocab = [x[0] for x in load_json(dir_path + "vocab.json")]
    cur = 0
    high_freq = set()
    for x in tqdm(vocab):
        if x not in stopwords:
            high_freq.add(x)
        if len(high_freq) > int(0.001 * len(vocab)):
            break
    save_txt("\n".join(list(high_freq)), rootdir + "high_freq.txt")
    for file in file_list:
        if ".txt" not in file:
            continue
        name = file[: file.rindex(".")]
        print(name)
        data = load_txt(dir_path + file)
        rate_all = 0
        line_n = len(data)
        pos = []
        for line in tqdm(data, mininterval=1):
            pos_one = []
            seq = line.strip().split()
            high_n = 0
            seq_n = len(seq)
            for word in seq:
                if word in stopwords:
                    pos_one.append("2")
                    seq_n -= 1
                elif word in high_freq:
                    pos_one.append("1")
                    high_n += 1
                else:
                    pos_one.append("0")
            assert len(pos_one) == len(seq)
            pos.append(" ".join(pos_one))
            if seq_n > 0:
                rate_all += high_n / seq_n
            else:
                line_n -= 1

        print(rate_all / line_n)
        assert len(data) == len(pos)
        save_txt("\n".join(pos), out_dir + "pos-" + name + ".txt")
        print(out_dir + "pos-" + name + ".txt")


def pos_onmt():
    stopwords = set(load_txt("./tool_data/stopwords.txt"))
    high_freq = set(load_txt("./tool_data/high_freq.txt"))
    stoi = load_json("./tool_data/stoi.json")
    vocab_onmt = [k for k in stoi.keys()]
    print(vocab_onmt[:10])
    stof = collections.OrderedDict()
    for x in vocab_onmt:
        if x in stopwords:
            stof[x] = 4
        elif x in high_freq:
            stof[x] = 5
        else:
            stof[x] = 6
    save_json(stof, "./tool_data/stof.json")


def freq_reddit(rootdir, out_dir):
    dir_path = rootdir + "data_reddit/"
    file_list = os.listdir(dir_path)
    # vocab = collections.Counter()
    # for file in file_list:
    #     if ".txt" not in file:
    #         continue
    #     name = file[: file.rindex(".")]
    #     print(name)
    #     data = load_txt(dir_path + file)
    #     for line in data:
    #         seq = line.strip().split()
    #         vocab.update(seq)
    # vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    # save_json(vocab, dir_path + "vocab.json")
    vocab = load_json(dir_path + "vocab.json")
    high_freq = set([x[0] for x in vocab[:int(0.005 * len(vocab))]])
    print(len(high_freq))
    # save_txt("\n".join(list(high_freq)), rootdir+"high_freq.txt")

    for file in file_list:
        if ".txt" not in file:
            continue
        name = file[: file.rindex(".")]
        print(name)
        data = load_txt(dir_path + file)
        rate_all = 0
        line_n = len(data)
        pos = []
        for line in tqdm(data, mininterval=1):
            pos_one = []
            seq = line.strip().split()
            high_n = 0
            seq_n = len(seq)
            for word in seq:
                if word in high_freq:
                    pos_one.append("1")
                    high_n += 1
                else:
                    pos_one.append("0")
            assert len(pos_one) == len(seq)
            pos.append(" ".join(pos_one))
            if seq_n > 0:
                rate_all += high_n / seq_n
            else:
                line_n -= 1

        print(rate_all / line_n)
        assert len(data) == len(pos)
        save_txt("\n".join(pos), out_dir + "tag-" + name + ".txt")
        print(out_dir + "tag-" + name + ".txt")


def freq_reddit_json(path, out_dir, high, n):
    data = load_json(path)
    if not os.path.exists(out_dir + "vocab.txt"):
        new_data = []
        vocab = collections.Counter()
        for dialog in tqdm(data):
            try:
                for seq in dialog:
                    assert len(seq) > 0
            except:
                continue
            new_dialog = []
            for seq in dialog:
                seq = seq.strip().lower()
                seq_list = nltk.word_tokenize(seq)
                new_dialog.append(" ".join(seq_list))
                vocab.update(seq_list)
            new_data.append(new_dialog)
        vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)[:n]
        print(len(vocab), "len vocab")
        vocab = [x[0] for x in vocab]
        save_txt("\n".join(vocab), out_dir + "vocab.txt")
        save_json(new_data, path)
        data = new_data
    else:
        vocab = load_txt(out_dir + "vocab.txt")
    print(len(vocab), "vocab")
    high_freq = set([x for x in vocab[:int(high * len(vocab))]])
    vocab = set([x for x in vocab])
    print(len(high_freq), "len high")
    # save_txt("\n".join(list(high_freq)), rootdir+"high_freq.txt")

    random.shuffle(data)

    test = data[-10000:]
    valid = data[-20000:-10000]
    train = data[:-20000]

    dataset = {"train": train, "valid": valid, "test": test}
    rate_all = 0
    line_n = 0
    for k, v in dataset.items():
        print(k)
        line_n += len(v)
        tag_data = []
        for dialog in tqdm(v, mininterval=1):
            tag_dialog = []
            for i, seq in enumerate(dialog):
                tag_seq = []
                high_n = 0
                seq_list = seq.split()
                seq_len = len(seq_list)
                for word in seq_list:
                    if (word in high_freq) or (word not in vocab):  # for unk
                        tag_seq.append("1")
                        if i != 0:
                            high_n += 1
                    else:
                        tag_seq.append("0")
                if i != 0:  # for eos
                    tag_seq.append("1")
                    high_n += 1
                    seq_len += 1
                tag_dialog.append(" ".join(tag_seq))

                if i != 0:
                    if seq_len > 0:
                        rate_all += high_n / seq_len
                    else:
                        import pdb
                        pdb.set_trace()
                        line_n -= 1
            tag_data.append(tag_dialog)
        assert len(v) == len(tag_data)
        src = [dialog[0] for dialog in v]
        tgt = [dialog[1] for dialog in v]
        save_txt("\n".join(src), out_dir + "src-" + k + ".txt")
        save_txt("\n".join(tgt), out_dir + "tgt-" + k + ".txt")
        tag_src = [dialog[0] for dialog in tag_data]
        tag_tgt = [dialog[1] for dialog in tag_data]
        save_txt("\n".join(tag_src), out_dir + "freq-src-" + k + ".txt")
        save_txt("\n".join(tag_tgt), out_dir + "freq-tgt-" + k + ".txt")
    print(line_n)
    print(rate_all / line_n)


def cnt_vocab(path, out_dir, high, n):
    data = load_json(path)
    vocab_tgt = collections.Counter()
    vocab_src = collections.Counter()
    for dialog in tqdm(data):
        src = dialog[0].strip().lower().split()
        vocab_src.update(src)
        tgt = dialog[1].strip().lower().split()
        vocab_tgt.update(tgt)
    vocab_src = sorted(vocab_src.items(), key=lambda x: x[1], reverse=True)
    print(len(vocab_src), "len vocab src")
    vocab_tgt = sorted(vocab_tgt.items(), key=lambda x: x[1], reverse=True)
    print(len(vocab_tgt), "len vocab tgt")
    save_json(vocab_src, out_dir + "vocab_src.json")
    save_json(vocab_tgt, out_dir + "vocab_tgt.json")

    high_freq = set([x[0] for x in vocab_tgt[:int(high * len(vocab_tgt))]])
    vocab_tgt = set([x[0] for x in vocab_tgt[:n]])
    print(len(high_freq), "len high")

    random.shuffle(data)

    test = data[-10000:]
    valid = data[-20000:-10000]
    train = data[:-20000]

    dataset = {"train": train, "valid": valid, "test": test}
    rate_all = 0
    line_n = 0
    for k, v in dataset.items():
        print(k)
        line_n += len(v)
        tag_data = []
        for dialog in tqdm(v, mininterval=1):
            tag_dialog = []
            for i, seq in enumerate(dialog):
                tag_seq = []
                high_n = 0
                seq_list = seq.split()
                seq_len = len(seq_list)
                for word in seq_list:
                    if (word in high_freq) or (word not in vocab_tgt):  # for unk
                        tag_seq.append("1")
                        if i != 0:
                            high_n += 1
                    else:
                        tag_seq.append("0")
                if i != 0:  # for eos
                    tag_seq.append("1")
                    high_n += 1
                    seq_len += 1
                tag_dialog.append(" ".join(tag_seq))

                if i != 0:
                    if seq_len > 0:
                        rate_all += high_n / seq_len
                    else:
                        import pdb
                        pdb.set_trace()
                        line_n -= 1
            tag_data.append(tag_dialog)
        assert len(v) == len(tag_data)
        # src = [dialog[0] for dialog in v]
        # tgt = [dialog[1] for dialog in v]
        # save_txt("\n".join(src), out_dir + "src-" + k + ".txt")
        # save_txt("\n".join(tgt), out_dir + "tgt-" + k + ".txt")
        # tag_src = [dialog[0] for dialog in tag_data]
        # tag_tgt = [dialog[1] for dialog in tag_data]
        # save_txt("\n".join(tag_src), out_dir + "freq-src-" + k + ".txt")
        # save_txt("\n".join(tag_tgt), out_dir + "freq-tgt-" + k + ".txt")
    print(line_n)
    print(rate_all / line_n)


def tri_reddit_json(path, out_dir, n):
    data = load_json(path)
    if not os.path.exists(out_dir + "vocab.txt"):
        new_data = []
        vocab = collections.Counter()
        for dialog in tqdm(data):
            try:
                for seq in dialog:
                    assert len(seq) > 0
            except:
                continue
            new_dialog = []
            for seq in dialog:
                seq = seq.strip().lower()
                seq_list = nltk.word_tokenize(seq)
                new_dialog.append(" ".join(seq_list))
                vocab.update(seq_list)
            new_data.append(new_dialog)
        vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)[:n]
        print(len(vocab), "len vocab")
        vocab = [x[0] for x in vocab]
        save_txt("\n".join(vocab), out_dir + "vocab.txt")
        save_json(new_data, path)
        data = new_data
    else:
        vocab = load_txt(out_dir + "vocab.txt")
    print(len(vocab), "vocab")

    # tagging
    vocab_pos = nltk.pos_tag(vocab)
    set_stopwords = set(stopwords.words('english'))
    set_vn = set()
    set_ord = set()

    stopwords_vocab = []
    vn_vocab = []
    ord_vocab = []
    itoj = [0, 1, 2, 3]
    tri_mask = {"stop": [False] * 50004, "vn": [False] * 50004, "ord": [False] * 50004}
    tri_mask["stop"][:4] = [True, True, True, True]

    for i, (word, pos) in enumerate(vocab_pos):
        if word in set_stopwords:
            itoj.append(len(stopwords_vocab) + 4)
            stopwords_vocab.append((word, i))
            tri_mask["stop"][i + 4] = True
        elif "V" in pos or "N" in pos:
            itoj.append(len(vn_vocab))
            vn_vocab.append((word, i))
            set_vn.add(word)
            tri_mask["vn"][i + 4] = True
        else:
            itoj.append(len(ord_vocab))
            ord_vocab.append((word, i))
            set_ord.add(word)
            tri_mask["ord"][i + 4] = True
    print(len(stopwords_vocab), "stop")
    print(len(vn_vocab), "vn")
    print(len(ord_vocab), "or")

    save_json(tri_mask, out_dir + "tri_mask.json")
    save_json(itoj, out_dir + "tri_itoj.json")
    save_json(stopwords_vocab, out_dir + "stop_vocab.json")
    save_json(vn_vocab, out_dir + "vn_vocab.json")
    save_json(ord_vocab, out_dir + "or_vocab.json")

    random.shuffle(data)

    test = data[-10000:]
    valid = data[-20000:-10000]
    train = data[:-20000]

    set_vocab = set(vocab)
    dataset = {"train": train, "valid": valid, "test": test}
    for k, v in dataset.items():
        print(k)
        tag_data = []
        for dialog in tqdm(v, mininterval=1):
            tag_dialog = []
            for i, seq in enumerate(dialog):
                seq_list = seq.strip().split()

                assert len(seq_list) > 0
                tag_seq = []
                for word in seq_list:
                    if (word in set_stopwords) or (word not in set_vocab):  # for unk
                        tag_seq.append("stop")
                    elif word in set_vn:
                        tag_seq.append("vn")
                    else:
                        tag_seq.append("ord")

                if i != 0:  # for eos
                    tag_seq.append("stop")
                tag_dialog.append(" ".join(tag_seq))
            tag_data.append(tag_dialog)

        assert len(v) == len(tag_data)
        if not os.path.exists(out_dir + "src-" + k + ".txt"):
            src = [dialog[0] for dialog in v]
            tgt = [dialog[1] for dialog in v]
            save_txt("\n".join(src), out_dir + "src-" + k + ".txt")
            save_txt("\n".join(tgt), out_dir + "tgt-" + k + ".txt")
        else:
            print("src exits")
        print(v[0])
        print(tag_data[0])
        tag_src = [dialog[0] for dialog in tag_data]
        tag_tgt = [dialog[1] for dialog in tag_data]
        save_txt("\n".join(tag_src), out_dir + "tri-src-" + k + ".txt")
        save_txt("\n".join(tag_tgt), out_dir + "tri-tgt-" + k + ".txt")


def main():
    # onmt_Daily("./train/dialogues_train.txt", "./onmt_data/", True)
    # onmt_Daily("./validation/dialogues_validation.txt", "./onmt_data/")
    # onmt_Daily("./test/dialogues_test.txt", "./onmt_data/")

    # onmt_reddit("/home/wangyida/data/reddit/200W_data.txt", "/home/wangyida/data/onmt_data/")
    # convert_vocab()
    # onmt_opensubtitle("/home/wangyida/git/OpenSubtitle/opensubtitles/", "/home/wangyida/git/OpenSubtitle/OpenNMT-py/data/")
    # freq_opensub("/home/wangyida/git/onmt_nlp/", "/home/wangyida/git/onmt_nlp/data/freq/")
    # convert_vocab()
    # pos_onmt()
    # freq_reddit("/home/wangyida/git/onmt_nlp/", "/home/wangyida/git/onmt_nlp/data_reddit/freq/")

    # freq_reddit_json("data_raw/reddit_small_single.json", "data_reddit_small/", 0.003, 50000)
    # cnt_vocab("data_raw/reddit_small_single.json", "data_reddit_small/", 0.001, 50000)

    tri_reddit_json("data_raw/reddit_small_single.json", "data_reddit_small/", 50000)
    print(1)


if __name__ == '__main__':
    main()
