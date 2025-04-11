# graphene phonon 

boundary p p p
units        metal
atom_style   atomic
box tilt large
timestep 0.0005
#neighbor 2.0 nsq
neigh_modify every 1 delay 5 check yes


#Gr.data
read_data    Gr_init.dat

mass  1 12.0107
 

pair_style   tersoff
pair_coeff   * * C.tersoff C

fix 	1 all box/relax x 0.0 y 0.0 vmax 0.001
min_style	cg
minimize     1.0e-12 1.0e-12 10000 10000

write_data   Gr.dat
