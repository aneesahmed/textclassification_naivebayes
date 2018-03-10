# coding=utf-8
def load_data():
    train_one = []
    train_zero =[]
    #labels = []
    # loading data into array
    labels = np.loadtxt("data/train_labels.txt", dtype=int)
    cnt = cl.Counter(labels)
    #print(cnt)
    #print(labels)
    f  = open("data/train.csv")
    csvreader = csv.reader(f)
    counter = 0
    for row in csvreader:
        #print(labels[counter])
        if labels[counter] == 1:
            #train_one.append([int(x) for x in row])
            train_one.append([int(x) for x in row])
        else:
            train_zero.append([int(x) for x in row])

        counter += 1

        #if counter > 10:
         #   break
    #print(len(train_one), len(train_zero))
    #print(sum(train_one[0]), sum(train_one[1649]), sum(train_one[1500]))
    train_1 = np.array(train_one)
    train_0 = np.array((train_zero))
    #print(train_1.shape, train_0.shape)
    return train_0, train_1, cnt # count 1 and 0 of the class