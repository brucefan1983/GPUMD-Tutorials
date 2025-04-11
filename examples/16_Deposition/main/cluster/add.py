import random

list1 = [0] ; list2 = [0] ; list3 = [0] ;  number = 1
for i in range(num):
    list1_1 = [] ; list2_1 = [] ; list3_1 = [] ; list4_1 = []
    a = 0 ; b = 0
    while a == 0:
        a = 1
        s1 = random.uniform(x_min,x0)
        s2 = random.uniform(y_min,y0)
        s3 = random.uniform(h_min,h_max)
        if s1 < cutoff :
            s1 = x0 + s1
        if s2 < cutoff :
            s2 = y0 + s2
        for j in range(len(list1)):
            x1 = (s1 - list1[j])**2 ; y1 = (s2 - list2[j])**2 ; z1 = (s3 - list3[j])**2
            if x1 + y1 + z1 <= cutoff**2 :
                a = 0 ; b = b + 1
                if b >= 1000:
                    print('Too many clusters were added in single cycle, please reduce the number value in \'parameter.in\'')
                    exit()
    if s1 > x0:
        s1 = s1 - x0
    if s2 > y0:
        s2 = s2 - y0
    list1.append(s1) ; list2.append(s2) ; list3.append(s3)
    file = open('./cluster/add%d.txt' % number,'r')
    lines = file.readlines()
    for line in lines:
        a = line.split()
        x = a[1] ; y = a[2] ; z = a[3] 
        x = float(x) ; y = float(y) ; z = float(z)
        x = x + s1 ; y = y + s2 ; z = z + s3
        v = 0 - velocity
        if x >= x0 :
            x = x - x0
        if y >= y0 :
            y = y - y0
        list1_1.append(x) ; list2_1.append(y)
        list3_1.append(z) ; list4_1.append(v)
    file.close()
    for i in range(len(list1_1)):
        lines[i] = ('%s %.4f %.4f %.4f %s 0 0 %.4f %s\n' % (a[0],list1_1[i],list2_1[i],list3_1[i],a[4],list4_1[i],group))
    file = open('./cluster/add%d.txt' % number,'w')
    for line in lines:
        file.writelines(line)
    file.close()
    number = number + 1
