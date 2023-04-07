s = input()
r = [int(item) for item in list(' '.join(s.split(',')))]
print(r)
while 0 in r:
    for i in range(len(r)):
        if r[i]==0:
            if i+1<len(r):
                r[i] = r[i+1]-1
            elif i-1>0:
                r[i] = r[i-1]+1
    print(r)
