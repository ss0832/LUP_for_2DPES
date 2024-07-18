import mb_pot
import derivatives
import lup

import matplotlib.pyplot as plt
import numpy as np


print("Locally Updated Plain (LUP) method for Muller-Brown potential")
init_point = np.array([[0.823499], [0.028038]]) #initial points
end_point = np.array([[0.158233], [0.501716]])
iteration = 40
number_of_nodes = 14

init_point_list = lup.lst(init_point, end_point, n_points=number_of_nodes)

# save Init condition
level = []
for i in range(-10, 40):
    level.append(15*i)
point_list = init_point_list
plt.plot(point_list.T[0][0], point_list.T[0][1], 'w.--')
x_list = np.linspace(-2.0, 1.5, 500)
y_list = np.linspace(-1.0, 2.5, 500)
plt.title('Iinitialization')
plt.xlabel('x')
plt.ylabel('y')
x_mesh, y_mesh = np.meshgrid(x_list, y_list)
f_mesh = mb_pot.muller_brown_potential(x_mesh, y_mesh)
cont = plt.contourf(x_mesh, y_mesh, f_mesh, levels=level, cmap='jet')
plt.colorbar()
plt.savefig('lup_init.png')
plt.close()
# -----

history = []

for j in range(iteration):
    point_list, energy_list = lup.lup(point_list)
    history.append([point_list, energy_list])
    print("Iteration: ", j)
    level = []
    for i in range(-10, 40):
        level.append(15*i)

    
    plt.plot(point_list.T[0][0], point_list.T[0][1], 'w.--')
    x_list = np.linspace(-2.0, 1.5, 500)
    y_list = np.linspace(-1.0, 2.5, 500)
    plt.title('Iteration: {}'.format(j))
    plt.xlabel('x')
    plt.ylabel('y')
    x_mesh, y_mesh = np.meshgrid(x_list, y_list)
    f_mesh = mb_pot.muller_brown_potential(x_mesh, y_mesh)
    cont = plt.contourf(x_mesh, y_mesh, f_mesh, levels=level, cmap='jet')
    plt.colorbar()
    plt.savefig('lup_{}.png'.format(j))
    plt.close()

print("LUP finished.")