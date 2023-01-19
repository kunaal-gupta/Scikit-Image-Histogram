n = int(input())
flag = 1/(n-1)
bin_len= [0]

for i in range(n-1):
    bin_len.append(bin_len[-1]+flag)
print(bin_len)

