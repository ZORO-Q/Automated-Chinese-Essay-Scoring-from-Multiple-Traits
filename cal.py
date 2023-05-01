f = open('data/data.json','r',encoding='utf-8')
lines = f.readlines()
lens = 0
sen_num = 0
for line in lines:
    line = eval(line)
    data = line['para']
    for item in data:
        for t in item:
            sen_num += 1
            lens += len(t[0])
print(lens)
print(sen_num)
print(lens/sen_num)
