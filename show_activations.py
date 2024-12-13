import json
import matplotlib.pyplot as plt
import numpy as np

def distance(alist, blist):
    sum_of = 0
    for x, y in zip(alist, blist):
        ans = (x - y)**2
        sum_of += ans
    return (sum_of)**(1/2)

file = "/fs/ess/scratch/PAS2099/jike/DelayBN/cifar_output/activs_log/SINGLE/delay0_shards.txt"
with open(file) as f:
    data = f.read()
data = json.loads(data)
l2_distance = []
x = [i*5 for i,_ in enumerate(data.keys()) if i!=0]

mean_true = []
mean_false = []
var_true = []
var_false = []
mean_true_false = []
mean_true_false_ = []
test = []
print("len", len(data.keys()))
for idx, (key, item) in enumerate(data.items()):
    if idx == 0:
        continue
    # print(idx, distance(item[0],item[1]))
    # l2_distance.append(distance(item[0],item[1]))
    mean_true.append(np.mean(item[0]))
    mean_false.append(np.mean(item[1]))
    # var_true.append(np.var(item[0]))
    # var_false.append(np.var(item[1]))
    # mean_true_false.append(np.mean([i-j for i,j in zip(item[0], item[1])]))
    # mean_true_false_.append(np.mean(item[0])-np.mean(item[1]))
    true_upper = np.percentile(item[0], 95)
    false_upper = np.percentile(item[1], 95)

    list1 = np.sort(item[0])
    list2 = np.sort(item[1])
    list1_95 = list1[-95]
    list2_95 = list2[-95]
    test.append(list1_95-list2_95)
    # if idx < len(data)-1:
    #     continue
    
    if idx != 100:
        continue
    # print(item[0][0:10])
    # print(item[1][0:10])

    # fig, ax = plt.subplots()
    # ax.plot(item[1], label="false stats")
    # ax.hlines(false_upper, 1, len(item[1]), color='g')
    # fig.savefig("figfalse.png")
    
    # plt.figure()
    # fig, ax = plt.subplots()
    # ax.plot(item[0], label="true stats")
    # ax.hlines(true_upper, 1, len(item[0]), color='g')
    # fig.savefig("figtrue.png")

    # raise
    # plt.ylim(0, 5)
    # plt.legend(loc="upper left")
    # plt.show()
    # plt.savefig("test.png")
    # plt.figure()
    
    # plt.savefig("test1.png")
    # raise
plt.plot(x, test)
z = np.polyfit(x, test, 2)
p = np.poly1d(z)
plt.plot(x, p(x))
plt.savefig("l2_distance.png")
raise
plt.figure(2)
# plt.plot(x,l2_distance)
plt.plot(x,mean_true, label="true stats")
plt.plot(x,mean_false, label="false stats")
plt.ylim(0, 0.5)
# plt.plot(x,mean_true_false)
# plt.plot(x,mean_true_false_)
plt.legend(loc="upper left")
# plt.title("L2 distance")
plt.xlabel("Round")
plt.ylabel("Var")
plt.show()
plt.savefig("l2_distance.png")
