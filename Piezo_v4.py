import numpy as np
from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
from pathlib import Path
from numba import njit
import os
import random as rdm
import time
import math

t1 = time.time()

eps = 10**(-4)
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
E_matr = 1e7  #Модуль Юнга для матрицы(в дин/см2)
########
#Множители для перевода в СИ:
P_transl = E_matr*33e-7
Sgm_transl = E_matr*0.1
H_transl = E_matr**(0.5)*1e-4
M_transl = E_matr**(0.5)*10**3/(4*np.pi)
lin_transl = (10*E_matr)**-1
#########
Nmesh_in = 5
Nmesh_ext = 50
alpha = 180
box_x = 1200
box_y = 2400
bbox = [[0, 0], [box_x, box_y]]
Ms = 0.01
angle = 0
R_piez = 100
R_mag = 30
gap = 50
d15 = 12e-6
d31 = -2e-6 #тут про минус забыл
d33 = 4.25e-6
H0_max = 15
Nmax = 30
G0_mag = 1
lam0_mag = 100
Ed=500
chi = 0
plot_flag=False

def round_sig(x, sig=4):
   return round(x, sig-int(math.floor(math.log10(abs(x))))-1)

@njit
def F_dd(mu1, mu2, r):  # = -grad U_dd
    nr = np.linalg.norm(r)
    return 3*(np.dot(mu1, mu2)*r + mu1*np.dot(mu2, r) + mu2*np.dot(mu1, r))/nr**5 - 15*np.dot(mu1, r)*np.dot(mu2, r)*r/nr**7

def calc_Fdd(f_list, rc_list, mu_list, vols):
    num_part = len(rc_list)
    for i in range(num_part-1):
        f_dd = np.zeros(2)
        for j in range(num_part-1):
            if i != j:
                r_ij = rc_list[i] - rc_list[j]
                f_dd += F_dd(mu_list[i], mu_list[j], r_ij)/vols[i]
        f_list[i].assign(Constant(f_dd))

def set_geom_modif(box_x = box_x, box_y = box_y, Ms = Ms, alpha = alpha, R_piez = R_piez, R_mag = R_mag, gap = gap, only_show = False):
    rc_dist = R_mag + R_piez + gap
    rc_list_mag = []
    rc_list_piez = []
    Ms_list = []
    x_1 = box_x/2 - rc_dist*np.sin(2*pi*alpha/(2*360))
    y_1 = box_x/2 - rc_dist*np.cos(2*pi*alpha/(2*360)) + (box_y-box_x)
    x_2 = box_x/2 + rc_dist*np.sin(2*pi*alpha/(2*360))
    y_2 = box_x/2 - rc_dist*np.cos(2*pi*alpha/(2*360)) + (box_y-box_x)
    rc_list_piez.append(np.array([box_x/2, box_x/2 + (box_y-box_x)]))
    rc_list_mag.append(np.array([x_1, y_1]))
    rc_list_mag.append(np.array([x_2, y_2]))
    
    alfa_mu_list = np.array([0, 0, 0])
    alfa_mu_list = (np.pi/4)*alfa_mu_list

    Ms_list = Ms*np.ones_like(alfa_mu_list)
    Ms_list[-1]=0*Ms_list[-1]

    if only_show:
        initial_plot(rc_list_mag, rc_list_piez, R_mag, R_piez, alfa_mu_list, save=False)
        exit()

    return alfa_mu_list, Ms_list, rc_list_mag, rc_list_piez

def initial_plot(rc_list_mag, rc_list_piez, R_mag, R_piez, alfa_mu_list, save_dir="", save=True):
    plt.figure()
    plt.axis("equal")
    tt = np.linspace(0, 2*np.pi, 201)
    for i in range(len(rc_list_mag)):
        dx = R_mag*np.sin(alfa_mu_list[i])
        dy = R_mag*np.cos(alfa_mu_list[i])
        plt.arrow(rc_list_mag[i][0]-dx/2, rc_list_mag[i][1]-dy/2, dx, dy, width=2)
        plt.plot(rc_list_mag[i][0] + R_mag*np.cos(tt), rc_list_mag[i][1] + R_mag*np.sin(tt), '-')
    for i in range(len(rc_list_piez)):
        plt.plot(rc_list_piez[i][0] + R_piez*np.cos(tt), rc_list_piez[i][1] + R_piez*np.sin(tt), '-')
    if save:
        plt.savefig(save_dir)
    else:
        plt.show()

def mesh_generator(mesh_params):
    geo = open("mesh_3circ_teml.geo").read()
    for par in mesh_params:
        geo = geo.replace(f"_{par}_", str(mesh_params[par]))
    os.makedirs(f"./meshes/", exist_ok=True)
    mesh_name = f"meshes/mesh_{os.getpid()}"

    ff = open(f"{mesh_name}.geo", "w")
    ff.write(geo)
    ff.close()
    os.system(f"gmsh -2 {mesh_name}.geo -format msh2")
    os.system(f"dolfin-convert {mesh_name}.msh {mesh_name}.xml")
    return mesh_name    

def calc(alfa_list, Ms_list, rc_list_mag, rc_list_piez, bbox=bbox,  chi=chi, H0_max=H0_max, 
            Nmax=Nmax, G0_mag=G0_mag, lam0_mag=lam0_mag, R_mag=R_mag, 
            R_piez=R_piez, Ed=Ed, plot_flag=plot_flag, d15 = d15, d31 = d31, d33 = d33, angle=angle, gap=gap, H0dir='x', conf_type="", main_dir=""):
    main_dir = Path(f"{main_folder}")
    calc_dir = Path(f"anti_sym_angle={angle}_H0max={H0_max}_Ms={Ms_list[-1]}_Rmag={R_mag}_gap={gap}")
    graf_dir = Path(f"Ms={Ms_list[-1]}_Rmag={R_mag}_gap={gap}")
    save_dir = f"./{main_dir}/{calc_dir}/{graf_dir}/"
    os.makedirs(f"./{main_dir}/{calc_dir}/", exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    initial_plot(rc_list_mag, rc_list_piez, R_mag, R_piez, alfa_mu_list, save_dir=save_dir+f"initial_state_angle={angle}_H0max={H0_max}_Ms={Ms_list[1]}_Rmag={R_mag}_gap={gap}.png")
    xdmf_file = XDMFFile(f"./{main_dir}/{calc_dir}/1.xdmf")
    xdmf_file.parameters["flush_output"] = True
    xdmf_file.parameters["functions_share_mesh"] = True
    xdmf_file.parameters["rewrite_function_mesh"] = False
    
    num_piez_part = len(rc_list_piez)
    num_mag_part = len(rc_list_mag)
    num_part = num_mag_part + num_piez_part

    mesh_params = {
    "lc_ext": Nmesh_ext, "lc_circ": Nmesh_in,
    "xmin": 0, "ymin": 0,
    "xmax": box_x, "ymax": box_y,
    "x1": rc_list_mag[0][0], "y1": rc_list_mag[0][1], "r1": R_mag,
    "x2": rc_list_mag[1][0], "y2": rc_list_mag[1][1], "r2": R_mag,
    "x3": rc_list_piez[0][0], "y3": rc_list_piez[0][1], "r3": R_piez,
     }
    #print(rc_list_mag, "\n")
    #print(rc_list_piez, "\n")
    #print(mesh_params)
    #exit()
    mesh_name = mesh_generator(mesh_params)
    mesh = Mesh(f'{mesh_name}.xml')
    subdom = MeshFunction("size_t", mesh, f"{mesh_name}_physical_region.xml")
    
    V0 = FunctionSpace(mesh, "DG", 0)
    subdom_f = Function(V0)
    ind_part = Function(V0)
    ind_matr = Function(V0)
    subdom_f.vector()[:] = subdom.array()[:]
    ind_part.vector()[:] = (subdom.array()[:]>=10).astype(int)
    ind_matr.vector()[:] = (subdom.array()[:]==0).astype(int)

    dx = Measure("dx", domain=mesh, subdomain_data=subdom)

    class Bottom(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] < (bbox[0][1] + 0.0001)

    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] < (bbox[0][0] + 0.0001)

    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] > (bbox[1][0] - 0.0001)

    class Up(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] > (bbox[1][1] - 0.0001)
    
    bottom = Bottom()
    left = Left()
    right = Right()
    up = Up()

    boundary_markers = MeshFunction("size_t", mesh, 1)
    up.mark(boundary_markers, 1)
    right.mark(boundary_markers, 2)

    V = VectorFunctionSpace(mesh, "CG", 1)
    Vt = TensorFunctionSpace(mesh, "DG", 0)

    bcs = [ 
          #DirichletBC(V.sub(1), Constant(0), up),
          DirichletBC(V, Constant([0, 0]), bottom),
          #DirichletBC(V.sub(0), Constant(0), left),
          #DirichletBC(V.sub(0), Constant(0), right)
        ]

    #G0_const_mag = Constant(G0_mag)
    #lmbda0_const_mag = Constant(lam0_mag)
    vH0 = Constant([0, 0])
    const_G0 = Constant(G0_mag) 
    const_k = Constant(Ed*G0_mag)
    
    du = TrialFunction(V)
    v = TestFunction(V)
    u = Function(V)
    u_old = Function(V)
    d = len(u)
    I = Identity(d)
    F = I + grad(u)
    C = F.T * F
    B = F * F.T
    J = det(F)
    E = 0.5*(C - I)
    e = sym(grad(u))
    Ic = J**(-2/3)*tr(C)

    sigma = const_G0*J**(-5/3)*B + (const_k*(J-1) - const_G0*Ic/(3*J))*I
    psi = const_G0/2*(Ic - 2)  +  0.5*const_k*(J - 1)**2
    Pi = psi * dx(0)

    f_list = []
    M0_list = []
    Pi += Ed*psi*dx(10)
    #for i in range(num_piez_part):
    #    Pi += Ed*psi*dx(10 + i)

    for i in range(num_mag_part):
        f_list.append(Constant([0, 0]))
        M0_list.append(Constant([Ms_list[i]*np.sin(alfa_list[i]), Ms_list[i]*np.cos(alfa_list[i])]))
        Pi += Ed*psi*dx(10 + num_piez_part + i) - inner(f_list[i], u)*dx(10 + num_piez_part + i) - inner(F*M0_list[i], vH0)*dx(10 + num_piez_part + i)

    FF = derivative(Pi, u, v)

    H_list = list(np.linspace(0, H0_max, Nmax))
    vols = [assemble(1*dx(10 + num_piez_part + i)) for i in range(num_part-1)]
    ########################
    H0 = []
    Sig_xx = []
    Sig_yy = []
    Sig_xy = []
    Px = []
    Py = []
    ff = open(f"./{main_dir}/{calc_dir}/{graf_dir}/piez_tension_angle={angle}_B0max={H0_max}_Ms={Ms_list[1]}_Rmag={R_mag}.txt", "a")
    ff.write("B()\tSig_xx()\tSig_xy()\tSig_yy()\n")
    ff.close()
    ff = open(f"./{main_dir}/{calc_dir}/{graf_dir}/piez_polaris_angle={angle}_B0max={H0_max}_Ms={Ms_list[1]}_Rmag={R_mag}.txt", "a")
    ff.write("B()\tPx()\tPy()\n")
    ff.close()
    ########################
    for h0 in H_list:
        print("h0=",h0)
        if conf_type != "max_magn_el_resp":
            if H0dir == "x":
                vH0.assign(Constant([h0, 0.]))
            if H0dir == "y":
                vH0.assign(Constant([0, h0]))
        else:
            vH0.assign(Constant([h0*np.sin(angle), h0*np.cos(angle)]))

        while True:
            F_pr = project(F, Vt)
            Mr_list = [F_pr(rc_list_mag[i]).reshape(2, 2).dot(M0_list[i].values()) for i in range(num_mag_part)]
            rc_list_curr = [rc_list_mag[i] + u(rc_list_mag[i]) for i in range(num_mag_part)]
            mu_list_curr = [(Mr_list[i] + chi*(vH0.values() - 4/3*pi*Mr_list[i])/(1 + 4/3*pi*chi))*vols[i] 
                             for i in range(num_mag_part)]
            calc_Fdd(f_list, rc_list_curr, mu_list_curr, vols)
            u_old.assign(u)
            solve(FF == 0, u, bcs, solver_parameters={"newton_solver": {"relative_tolerance": 1e-6, "absolute_tolerance":1e-7}})
            if assemble(inner(u - u_old, u - u_old)*dx)/assemble(1*dx) < eps:
                print("Iteration finished.")
                break
            print("eps=",assemble(inner(u - u_old, u - u_old)*dx))
        
        sig_xx = Ed*assemble(sigma[0, 0]*dx(10))/assemble(1*dx(10))
        sig_xy = Ed*assemble(sigma[0, 1]*dx(10))/assemble(1*dx(10))
        sig_yy = Ed*assemble(sigma[1, 1]*dx(10))/assemble(1*dx(10))
        R = 0.5*d15
        Q = d31
        S = d33 - d15 - d31
        #главная ось пьезо направлена по оси У
        px = 2*R*sig_xy
        py = Q*sig_xx + (Q + 2*R + S)*sig_yy
        H0.append(h0)
        Px.append(px)
        Py.append(py)
        Sig_xx.append(sig_xx)
        Sig_xy.append(sig_xy)
        Sig_yy.append(sig_yy)
        ff = open(f"./{main_dir}/{calc_dir}/{graf_dir}/piez_tension_angle={angle}_B0max={H0_max}_Ms={Ms_list[1]}_Rmag={R_mag}.txt", "a")
        ff.write(f"{h0*H_transl}\t{sig_xx*Sgm_transl}\t{sig_xy*Sgm_transl}\t{sig_yy*Sgm_transl}\n")
        ff.close()
        ff = open(f"./{main_dir}/{calc_dir}/{graf_dir}/piez_polaris_angle={angle}_B0max={H0_max}_Ms={Ms_list[1]}_Rmag={R_mag}.txt", "a")
        ff.write(f"{h0*H_transl}\t{px*P_transl}\t{py*P_transl}\n")
        ff.close()
        u.rename("u", "")
        subdom_f.rename("subdom", "")
        Epr = project(E)
        Epr.rename("E", "")
        xdmf_file.write(u, h0)
        xdmf_file.write(subdom_f, h0)
        xdmf_file.write(Epr, h0)
        #plt.figure()
        #plot(u, mode='displacement')
        #plt.show()
    #plt.figure()
    #plt.plot(H0, Sig_xx)
    #plt.plot(H0, Sig_xy)
    #plt.plot(H0, Sig_yy)
    
    """plt.figure(clear=True)
    plt.plot(H0, Px, label="Px")
    plt.plot(H0, Py, label="Py")
    plt.xlabel("Внешнее поле, у.е.")
    plt.ylabel("Поляризация, у.е.")
    plt.title(f"angle={angle}_Ms={Ms_list[1]*M_transl}A/m_gap={gap*lin_transl}m\nRmag={R_mag*lin_transl}m_Rpiex={R_piez*lin_transl}m")
    plt.legend()
    plt.savefig(f"./{main_dir}/{calc_dir}/{graf_dir}/piez_polariz_angle={angle}_B0max={H0_max}Тл_Ms={Ms_list[1]}A/m_Rmag={R_mag}m_gap={gap}m.png")"""
    return Px[-1], Py[-1], Sig_xx[-1], Sig_xy[-1], Sig_yy[-1]

#for GAP in [gap]:
#    for MS in [1]:
#        for R_MAG in [0.2]:
#            alfa_mu_list, Ms_list, rc_list_mag, rc_list_piez = set_geom(angle=180, Ms=MS, R_mag=R_MAG, gap=GAP)
#            calc(alfa_list=alfa_mu_list, Ms_list = Ms_list, rc_list_mag=rc_list_mag, rc_list_piez=rc_list_piez, angle=180, R_mag=R_MAG, gap=GAP)


main_folder = "dif_R"
conf_type="2part"
RR_mag=np.linspace(10, 60, 20)
Px = []
Py=[]
S_xx=[]
S_xy=[]
S_yy=[]
for i in RR_mag:
    ff = open(f"{main_folder}({i})_dimensionless_S.txt", "a")
    ff.write(f"Sig_xx\tSig_xy\tSig_yy\n")
    ff.close()
ff = open(f"{main_folder}({RR_mag[0]}-{RR_mag[-1]})_P.txt", "a")
ff.write(f"gap\tPx(Кл/м2)\tPy(Кл/м2)\n")
ff.close()
ff = open(f"{main_folder}({RR_mag[0]}-{RR_mag[-1]})_S.txt", "a")
ff.write(f"gap\tSig_xx(Па)\tSig_xy(Па)\tSig_yy(Па)\n")
ff.close()
for i in RR_mag:
    alfa_mu_list, Ms_list, rc_list_mag, rc_list_piez = set_geom_modif(alpha=180, R_mag=i)
    px, py, sig_xx, sig_xy, sig_yy = calc(R_mag = i, alfa_list=alfa_mu_list, Ms_list = Ms_list, rc_list_mag=rc_list_mag, rc_list_piez=rc_list_piez, angle=180, main_dir=main_folder, conf_type = conf_type)
    ff = open(f"{main_folder}({i})_dimensionless_S.txt", "a")
    ff.write(f"{sig_xx}\t{sig_xy}\t{sig_yy}\n")
    ff.close()
    px*=P_transl
    py*=P_transl
    sig_xx*=Sgm_transl
    sig_xy*=Sgm_transl
    sig_yy*=Sgm_transl
    Px.append(px)
    Py.append(py)
    S_xx.append(sig_xx)
    S_xy.append(sig_xy)
    S_yy.append(sig_yy)
    ff = open(f"{main_folder}({RR_mag[0]}-{RR_mag[-1]})_P.txt", "a")
    ff.write(f"{i*lin_transl}\t{px}\t{py}\n")
    ff.close()
    ff = open(f"{main_folder}({RR_mag[0]}-{RR_mag[-1]})_S.txt", "a")
    ff.write(f"{i*lin_transl}\t{sig_xx}\t{sig_xy}\t{sig_yy}\n")
    ff.close()
H0_max*=H_transl
plt.figure(clear=True)
plt.plot(gap, Px, label=f"Px(H_max={round_sig(H0_max, 3)}Тл)")
plt.plot(gap, Py, label=f"Py(H_max={round_sig(H0_max, 3)}Тл)")
plt.xlabel("Зазор между пьезо и магнитной частицами, у.е.")
plt.ylabel("Поляризация, Кл/м2")
plt.legend()
plt.savefig(f"{main_folder}({RR_mag[0]}-{RR_mag[-1]})P.png")
plt.figure(clear=True)
plt.plot(gap, Px, label=f"Px(H_max={round_sig(H0_max, 3)}Тл)")
plt.xlabel("Зазор между пьезо и магнитной частицами, у.е.")
plt.ylabel("Поляризация, Кл/м2")
plt.legend()
plt.savefig(f"{main_folder}({RR_mag[0]}-{RR_mag[-1]})_Px.png")
plt.figure(clear=True)
plt.plot(gap, Py, label=f"Py(H_max={round_sig(H0_max, 3)}Тл)")
plt.xlabel("Зазор между пьезо и магнитной частицами, у.е.")
plt.ylabel("Поляризация, Кл/м2")
plt.legend()
plt.savefig(f"{main_folder}({RR_mag[0]}-{RR_mag[-1]})_Py.png")
plt.figure(clear=True)

plt.plot(gap, S_xx, label=f"Sigm_xx(H_max={round_sig(H0_max, 3)}Тл)")
plt.plot(gap, S_xy, label=f"Sigm_xy(H_max={round_sig(H0_max, 3)}Тл)")
plt.plot(gap, S_yy, label=f"Sigm_yy(H_max={round_sig(H0_max, 3)}Тл)")
plt.xlabel("Зазор между пьезо и магнитной частицами, у.е.")
plt.ylabel("Напряжение, Па")
plt.legend()
plt.savefig(f"{main_folder}({RR_mag[0]}-{RR_mag[-1]})S.png")
plt.figure(clear=True)

t2 = time.time()
print("run time:", t2-t1, "sec")