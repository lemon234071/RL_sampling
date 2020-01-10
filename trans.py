import os

index = 0
for i in range(20):
    j = (i + 1) / 10
    cmd = "cat ./freq0.001/pos-src-train.txt | head -n {:d} | tail -n +{:d} > t_resp/{:.1f}pos-src-train.txt".format(
        index + 50000, index, j)
    print(cmd)
    os.system(cmd)
    index += 50001

for i in range(20):
    j = (i + 1) / 10
    cmd = "python3 translate.py -gpu 0 -model ./checkpoint/freq_2gen_step_40000.pt -output ./result/t_resp/infer{:.1f}.txt -beam 1 -batch_size 1024 -src ./data/t_resp/{:.1f}src-train.txt -pos_src ./data/t_resp/{:.1f}pos-src-train.txt -max_length 30".format(
        j, j, j)
    print(cmd)
    os.system(cmd)
    print(i, "over")
