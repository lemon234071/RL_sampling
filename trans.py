import os

index = 0
for i in range(20):
    j = (i + 1) / 10
    cmd = "cat ./data/freq0.001/pos-src-train.txt | head -n {:d} | tail -n +{:d} > ./data/t_resp/{:.1f}pos-src-train.txt".format(
        index + 50000, index, j)
    # print(cmd)
    # os.system(cmd)
    index += 50001

for file in ["valid", "test"]:
    for i in range(20):
        j = (i + 1) / 10
        # cmd = "CUDA_VISIBLE_DEVICES=2 python3 translate.py -gpu 0 -model ./checkpoint/freq_2gen_step_40000.pt -output ./data/t_resp/infer{:.1f}.txt -beam 1 -batch_size 2048 -src ./data/t_resp/{:.1f}src-train.txt -pos_src ./data/t_resp/{:.1f}pos-src-train.txt -max_length 30 -learned_t {:.1f}".format(
        #     j, j, j, j)
        cmd = "CUDA_VISIBLE_DEVICES=2 python3 translate.py -gpu 0 -model ./checkpoint/freq_2gen_step_40000.pt -output ./data/t_resp/{}{:.1f}.txt -beam 1 -batch_size 2048 -src ./data/src-{}.txt -pos_src ./data/freq0.001/pos-src-{}.txt -max_length 30 -learned_t {:.1f}".format(
            file, j, file, file, j)
        print(cmd)
        os.system(cmd)
        print(i, "over")
