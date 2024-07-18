import numpy as np
import mb_pot
import derivatives

# ref.1: https://superadditive.com/notes/cecam-molecular-kinetics-2016/locally-updated-planes.html
# ref.2: Ulitsky, A., & Elber, R. (1990). A new technique to calculate steepest descent paths in flexible polyatomic systems. The Journal of Chemical Physics, 92(2), 1510. doi:10.1063/1.458112
# ref.3: Müller, K. (1980). Reaction paths on multidimensional energy hypersurfaces. Angewandte Chemie International Edition in English, 19(1), 1–13. doi:10.1002/anie.198000013
 
def lst(init_point_1, init_point_2, n_points):# for initialization of the path
    #linear synchronous transit method
    #ref.: T.A. Halgren, W. N. Lipscomb, Chem. Phys. Lett. 49, 225 (1977)
    point_list = []
    for i in range(n_points + 1):
        point_list.append(init_point_1 + i * (init_point_2 - init_point_1) / n_points)
    point_list.append(init_point_2)
    point_list = np.array(point_list)
    return point_list # length:n_points + 2

def calc_tangent_vec(point1, point2):
    return (point2 - point1) / np.linalg.norm(point2 - point1)

def project_tangent_vec_for_grad(grad, tangent):
    proj_grad = grad - (grad * tangent) * tangent
    return proj_grad

def project_tangent_vec_for_hess(hess, tangent):
    ones_mat = np.eye(2)
    L_mat = tangent
    E_LL = ones_mat -1* np.dot(L_mat, L_mat.T)
    
    hess_proj = np.dot(np.dot(E_LL, hess), E_LL)
    return hess_proj


def lup(point_list, alpha=3e-4):
    new_point_list = []
    new_point_energy_list = []
    
    for i in range(len(point_list)):
        point = point_list[i]
        print("# NODE", i)
        if i == 0 or i == len(point_list)-1:
            grad = derivatives.grad(mb_pot.muller_brown_potential, point[0][0], point[1][0])
            
            new_point = point - alpha * grad
           
            new_point_list.append(new_point)
            new_point_energy_list.append(mb_pot.muller_brown_potential(new_point[0][0], new_point[1][0]))
        else:
            grad = derivatives.grad(mb_pot.muller_brown_potential, point[0][0], point[1][0])
            hess = derivatives.hess(mb_pot.muller_brown_potential, point[0][0], point[1][0])
            tangent = calc_tangent_vec(point_list[i-1], point_list[i+1])
            proj_grad = project_tangent_vec_for_grad(grad, tangent)
            proj_hess = project_tangent_vec_for_hess(hess, tangent)
            move_vec = -1 * np.dot(np.linalg.pinv(proj_hess), proj_grad)
            
            trust_radius = min(np.linalg.norm(point - point_list[i-1]), np.linalg.norm(point_list[i+1] - point)) / 2
            move_vec = move_vec * min(trust_radius, np.linalg.norm(move_vec)) / np.linalg.norm(move_vec)
            new_point = point + move_vec
           
            new_point_list.append(new_point)
            new_point_energy_list.append(mb_pot.muller_brown_potential(new_point[0][0], new_point[1][0]))
    new_point_list = np.array(new_point_list)
    new_point_energy_list = np.array(new_point_energy_list)
   
    return new_point_list, new_point_energy_list
    
