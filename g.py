a = {1:2, 2:4, 3:6}

prev = 0
for i in a:
    a[i] = a[i] +prev
    prev = a[i]

print(a)