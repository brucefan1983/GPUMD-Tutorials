#read information of simulation box
file = open('./model.xyz','r')
lines = file.readlines() ; num = 0
for line in lines:
    num = num + 1
    if num == 1:
        atom_num = int(line)
    elif num == 2:
        a = line.split("Lattice=") ; b = a[1].split("\"") ; c = b[1].split(" ")
        x = float(c[0]) ; y = float(c[4])
        print('Box x-direction length: %f' % x)
        print('Box y-direction length: %f' % y)
        break
file.close()

# defult parameters
cycle = 5 ; number = 15 ; velocity = 0.005 ; cutoff = 7 ; x0 = 0 ; y0 = 0

#read information of basic parameters for .sh
file = open('./parameter.in','r')
lines = file.readlines()
for line in lines:
    key = line.split()
    if key != []:
        if key[0] == 'path':
            path = key[1]
        elif key[0] == 'cycle':
            cycle = key[1]
        elif key[0] == 'species':
            species = key[1]
        elif key[0] == 'number':
            number = key[1]
        elif key[0] == 'velocity':
            velocity = key[1]
        elif key[0] == 'cutoff':
            cutoff = key[1]
        elif key[0] == 'h_min':
            h_min = key[1]
        elif key[0] == 'h_max':
            h_max = key[1]
        elif key[0] == 'x_max':
            x = key[1]
        elif key[0] == 'y_max':
            y = key[1]
        elif key[0] == 'x_min':
            x0 = key[1]
        elif key[0] == 'y_min':
            y0 = key[1]
        elif key[0] == 'h_cutoff':
            h_c = key[1]
        elif key[0] == 'group':
            group = key[1]
print('Total cycle : %s' % cycle)
print('Total kinds of clusters : %s' % species)  
print('Total clusters of every step : %s' % number)  
print('Deposition velocity : %s' % velocity) 
print('Minimum distance between two clusters : %s' % cutoff)
print('Starting height of deposition from %s to %s (at z direction)' % (h_min,h_max))
print('Starting height of deposition from %s to %s (at x direction)' % (x0,x))
print('Starting height of deposition from %s to %s (at y direction)' % (y0,y))
file.close()
#modify submit.sh
file = open('./main.sh','r')
lines = file.readlines()
file.close()
file = open('./main.sh','w')
for line in lines:
    if 'define' in line:
        file.writelines('#define parameters\ncycle=%s\nspecies=%s\nnumber=%s\n' % (cycle,species,number))
    elif 'out' in line:
        file.writelines('    out=`%s > ./output`\n' % path)
    else:
        file.writelines(line)
file.close()

#modify add.py
file = open('./cluster/add.py','r')
lines = file.readlines()
file.close()
file = open('./cluster/add.py','w')
file.writelines('x0 = %s ; y0 = %s ; x_min = %s ; y_min = %s\nvelocity = %s ; h_min = %s ; h_max = %s ; cutoff = %s ; num = %s ; group = %s\n' % (x,y,x0,y0,velocity,h_min,h_max,cutoff,number,group))
for line in lines:
    file.writelines(line)
file.close()

#modify z.py
file = open('./cluster/z.py','r')
lines = file.readlines()
file.close()
file = open('./cluster/z.py','w')
file.writelines('number = %s ; h_c = %s\n' % (atom_num,h_c))
for line in lines:
    file.writelines(line)
file.close()
