file = open('./restart.xyz','r')
lines = file.readlines()
b = 0
list1 = []
for line in lines:
    b = b + 1
    if b > 3:
        a = line.split()
        z = a[3] ; z = float(z)
        if z < h_c:
            list1.append(line)
    else:
        list1.append(line)       
file.close()
file = open('./restart.xyz','w')
for line in list1:
	file.writelines(line)
file.close()
c = len(lines) - 2
d = len(list1) - 2
e = c - d
list2 = ['Existing atomic number = %d\n' % d,'Deleted atomic number = %d\n' % e]
new_num = d - number
file = open('./cluster/add.txt','r')
num = 0
lines = file.readlines()
for line in lines:
    num = num + 1
file.close()
added = num - e
print('Number of new atoms added in this cycle : %s' % num)
print('Number of new atoms saved in this cycle : %s' % added)
print('Number of new atoms saved in total : %s' % new_num)
file = open('./delete.txt','w')
for line in list2:
	file.writelines(line)
file.close()