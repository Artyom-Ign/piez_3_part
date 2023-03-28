import numpy as np
from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
from pathlib import Path
from numba import njit
import os
import random as rdm
import time

t1 = time.time()

eps = 10**(-4)
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"

conf_type="max_magn_el_resp"
box_x = 5
box_y = 10
bbox = [[0, 0], [box_x, box_y]]
Ms = 0.1 # , 0.5, 1]:
angle = 0
R_piez = 1
R_mag = 0.2
gap = 0.1
d15 = 12e-6
d31 = 2e-6
d33 = 4.25e-6
Nmesh = 200
H0_max = 0.15
Nmax = 3
G0_mag = 1
lam0_mag = 100
G0_piez=1
lam0_piez=100
Ed=500
E_piez = 115.0
nu_piez = 0.3
chi = 0
plot_flag=False
unit_vect = Constant([0.0, 1.0])
lmbda = Constant(E_piez*nu_piez/((1 + nu_piez)*(1 - 2*nu_piez)))

def plot_graf(rc_list, Ms_list, R_list, angle, H0_max, gap, conf):
    plt.figure()
    plt.axis("equal")
    tt = np.linspace(0, 2*np.pi, 201)
    for i in range(len(rc_list)):
        dxx = R_list[i]/2*Ms_list[i][0]
        dyy = R_list[i]/2*Ms_list[i][1]
        plt.arrow(rc_list[i][0], rc_list[i][1], dxx, dyy, width=0.1)
        plt.plot(rc_list[i][0] + R_list[i]*np.cos(tt), rc_list[i][1] + R_list[i]*np.sin(tt), '-')
    plt.savefig(f"./{conf}_el_polarization_angle={angle}_H0max={H0_max}_Ms={Ms_list[1]}_Rmag={R_mag}_Rpiez={R_piez}_gap={gap}.png")
    plt.figure(clear = True)

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

def rand_distr(box_x = box_x, box_y = box_y, conf_type = "angle", Ms = Ms, R_piez = R_piez, R_mag = R_mag, angle = angle, only_show = False):
    rc_list_mag = []
    rc_list_piez = []
    Ms_list = []
    rc_x = [1, 5, 9, 3, 5, 7, 2.3, 9, 1, 8, 3, 5, 9, 6.5, 1, 4]
    rc_y = [9, 9, 9, 8, 7.5, 7, 6, 6, 4, 4.5, 3, 2.9, 2.9, 2, 2, 1]
    rc_list_piez.append(np.array([box_x/2, box_y/2]))
    for i in range(len(rc_x)):
        rc_list_mag.append(np.array([rc_x[i], rc_y[i]]))
    
    if conf_type == "all_up":
        alfa_mu_list = np.zeros((len(rc_list_mag)+1))
        alfa_mu_list = (np.pi/4)*alfa_mu_list

    if conf_type == "random":
        alfa_mu_list = np.random.rand(len(rc_list_mag)+1)*8
        alfa_mu_list = (np.pi/4)*alfa_mu_list

    if conf_type == "angle":
        alfa_mu_list = np.ones((len(rc_list_mag)+1))
        alfa_mu_list = (np.pi/180)*alfa_mu_list*angle
    Ms_list = Ms*np.ones_like(alfa_mu_list)
    Ms_list[-1]=0
    plt.figure()
    plt.axis("equal")
    tt = np.linspace(0, 2*np.pi, 201)
    for i in range(len(rc_list_mag)):
        dx = 2*R_mag*np.sin(alfa_mu_list[i])
        dy = 2*R_mag*np.cos(alfa_mu_list[i])
        plt.arrow(rc_list_mag[i][0], rc_list_mag[i][1], dx, dy, width=0.1)
        plt.plot(rc_list_mag[i][0] + R_mag*np.cos(tt), rc_list_mag[i][1] + R_mag*np.sin(tt), '-')
    for i in range(len(rc_list_piez)):
        #dx = 2*Ms*np.sin(alfa_mu_list[i])
        #dy = 2*Ms*np.cos(alfa_mu_list[i])
        #plt.arrow(rc_list_piez[i][0], rc_list_piez[i][1], dx, dy, width=0.1)
        plt.plot(rc_list_piez[i][0] + R_piez*np.cos(tt), rc_list_piez[i][1] + R_piez*np.sin(tt), '-')
    #plt.show()
    if only_show:
        plt.show()
        exit()

    return alfa_mu_list, Ms_list, rc_list_mag, rc_list_piez    

def set_geom(box_x = box_x, box_y = box_y, conf_type = conf_type, Ms = Ms, angle = angle, R_piez = R_piez, R_mag = R_mag, gap = gap, only_show = False):
    rc_dist = R_mag + R_piez + gap
    rc_list_mag = []
    rc_list_piez = []
    Ms_list = []
    x_1 = box_x/2 - rc_dist*np.sin(2*pi*angle/(2*360))
    y_1 = box_y/2 - rc_dist*np.cos(2*pi*angle/(2*360))
    x_2 = box_x/2 + rc_dist*np.sin(2*pi*angle/(2*360))
    y_2 = box_y/2 - rc_dist*np.cos(2*pi*angle/(2*360))
    rc_list_piez.append(np.array([box_x/2, box_y/2]))
    rc_list_mag.append(np.array([x_1, y_1]))
    rc_list_mag.append(np.array([x_2, y_2]))
    if conf_type=="max_magn_el_resp":
        rc_list_mag = []
        rc_list_piez = []
        Ms_list = []
        x_1 = box_x/2 + rc_dist*np.sin(2*pi*angle/(360))
        y_1 = box_y/2 + rc_dist*np.cos(2*pi*angle/(360))
        x_2 = box_x/2 - rc_dist*np.sin(2*pi*angle/(360))
        y_2 = box_y/2 - rc_dist*np.cos(2*pi*angle/(360))
        rc_list_piez.append(np.array([box_x/2, box_y/2]))
        rc_list_mag.append(np.array([x_1, y_1]))
        rc_list_mag.append(np.array([x_2, y_2]))
        alfa_mu_list = np.ones((len(rc_list_mag)+1))
        alfa_mu_list = (np.pi/180)*alfa_mu_list*(angle+90)

    if conf_type == "2part":
        alfa_mu_list = np.array([0, 0, 0])
        alfa_mu_list = (np.pi/4)*alfa_mu_list

    Ms_list = Ms*np.ones_like(alfa_mu_list)
    Ms_list[-1]=0*Ms_list[-1]

    if only_show:
        initial_plot(rc_list_mag, rc_list_piez, R_mag, R_piez, alfa_mu_list, save=False)
        exit()

    return alfa_mu_list, Ms_list, rc_list_mag, rc_list_piez

def set_geom_modif(box_x = box_x, box_y = box_y, conf_type = conf_type, Ms = Ms, angle = angle, R_piez = R_piez, R_mag = R_mag, gap = gap, only_show = False):
    rc_dist = R_mag + R_piez + gap
    rc_list_mag = []
    rc_list_piez = []
    Ms_list = []
    x_1 = box_x/2 - rc_dist*np.sin(2*pi*angle/(2*360))
    y_1 = box_x/2 - rc_dist*np.cos(2*pi*angle/(2*360)) + (box_y-box_x)
    x_2 = box_x/2 + rc_dist*np.sin(2*pi*angle/(2*360))
    y_2 = box_x/2 - rc_dist*np.cos(2*pi*angle/(2*360)) + (box_y-box_x)
    rc_list_piez.append(np.array([box_x/2, box_x/2 + (box_y-box_x)]))
    rc_list_mag.append(np.array([x_1, y_1]))
    rc_list_mag.append(np.array([x_2, y_2]))
    if conf_type=="max_magn_el_resp":
        rc_list_mag = []
        rc_list_piez = []
        Ms_list = []
        x_1 = box_x/2 + rc_dist*np.sin(2*pi*angle/(360))
        y_1 = box_x/2 + rc_dist*np.cos(2*pi*angle/(360)) + (box_y-box_x)
        x_2 = box_x/2 - rc_dist*np.sin(2*pi*angle/(360))
        y_2 = box_x/2 - rc_dist*np.cos(2*pi*angle/(360)) + (box_y-box_x)
        rc_list_piez.append(np.array([box_x/2, box_x/2 + (box_y-box_x)]))
        rc_list_mag.append(np.array([x_1, y_1]))
        rc_list_mag.append(np.array([x_2, y_2]))
        alfa_mu_list = np.ones((len(rc_list_mag)+1))
        alfa_mu_list = (np.pi/180)*alfa_mu_list*(angle+90)

    if conf_type == "2part":
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
        dx = 2*R_mag*np.sin(alfa_mu_list[i])
        dy = 2*R_mag*np.cos(alfa_mu_list[i])
        plt.arrow(rc_list_mag[i][0], rc_list_mag[i][1], dx, dy, width=0.1)
        plt.plot(rc_list_mag[i][0] + R_mag*np.cos(tt), rc_list_mag[i][1] + R_mag*np.sin(tt), '-')
    for i in range(len(rc_list_piez)):
        plt.plot(rc_list_piez[i][0] + R_piez*np.cos(tt), rc_list_piez[i][1] + R_piez*np.sin(tt), '-')
    if save:
        plt.savefig(save_dir)
    else:
        plt.show()

def calc(alfa_list, Ms_list, rc_list_mag, rc_list_piez, bbox=bbox,  chi=chi, Nmesh=Nmesh, H0_max=H0_max, 
            Nmax=Nmax, G0_mag=G0_mag, lam0_mag=lam0_mag, R_mag=R_mag, 
            R_piez=R_piez, Ed=Ed, plot_flag=plot_flag, d15 = d15, d31 = d31, d33 = d33, angle=angle, gap=gap, H0dir='x', conf_type="", main_dir=""):
    main_dir = Path(f"{main_folder}")
    calc_dir = Path(f"anti_sym_angle={angle}_H0max={H0_max}_Ms={Ms_list[-1]}_Rmag={R_mag}_gap={gap}")
    graf_dir = Path(f"Ms={Ms_list[-1]}_Rmag={R_mag}_gap={gap}")
    save_dir = f"./{main_dir}/{calc_dir}/{graf_dir}/"
    os.makedirs(save_dir, exist_ok=True)
    initial_plot(rc_list_mag, rc_list_piez, R_mag, R_piez, alfa_mu_list, save_dir=save_dir+f"initial_state_angle={angle}_H0max={H0_max}_Ms={Ms_list[1]}_Rmag={R_mag}_gap={gap}.png")
    xdmf_file = XDMFFile(f"./{main_dir}/{calc_dir}/1.xdmf")
    xdmf_file.parameters["flush_output"] = True
    xdmf_file.parameters["functions_share_mesh"] = True
    xdmf_file.parameters["rewrite_function_mesh"] = False
    
    num_piez_part = len(rc_list_piez)
    num_mag_part = len(rc_list_mag)
    num_part = num_mag_part + num_piez_part

    domain = Rectangle(Point(bbox[0]), Point(bbox[1]))

    for (i, rc) in enumerate(rc_list_piez):
        cir = Circle(Point(rc), R_piez) 
        domain.set_subdomain(10 + i, cir)
        print("##############\n", 10 + i, "\n##############")

    for (i, rc) in enumerate(rc_list_mag):
        cir = Circle(Point(rc), R_mag)
        domain.set_subdomain(10 + i + num_piez_part, cir)

    mesh = generate_mesh(domain, Nmesh)
    subdom = MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.domains())
    
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
    ff = open(f"./{main_dir}/{calc_dir}/{graf_dir}/piez_tension_angle={angle}_H0max={H0_max}_Ms={Ms_list[1]}_Rmag={R_mag}.txt", "a")
    ff.write(f"H\tSig_xx\tSig_xy\tSig_yy\n")
    ff.close()
    ff = open(f"./{main_dir}/{calc_dir}/{graf_dir}/piez_polaris_angle={angle}_H0max={H0_max}_Ms={Ms_list[1]}_Rmag={R_mag}.txt", "a")
    ff.write(f"H\tPx\tPy\n")
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
###        
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
        ff = open(f"./{main_dir}/{calc_dir}/{graf_dir}/piez_tension_angle={angle}_H0max={H0_max}_Ms={Ms_list[1]}_Rmag={R_mag}.txt", "a")
        ff.write(f"{h0}\t{sig_xx}\t{sig_xy}\t{sig_yy}\n")
        ff.close()
        ff = open(f"./{main_dir}/{calc_dir}/{graf_dir}/piez_polaris_angle={angle}_H0max={H0_max}_Ms={Ms_list[1]}_Rmag={R_mag}.txt", "a")
        ff.write(f"{h0}\t{px}\t{py}\n")
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
    
    plt.figure(clear=True)
    plt.plot(H0, Px, label="Px")
    plt.plot(H0, Py, label="Py")
    plt.xlabel("Внешнее поле, у.е.")
    plt.ylabel("Поляризация, у.е.")
    plt.title(f"angle={angle}_Ms={Ms_list[1]}_gap={gap}\nRmag={R_mag}_Rpiex={R_piez}")
    plt.legend()
    plt.savefig(f"./{main_dir}/{calc_dir}/{graf_dir}/piez_polariz_angle={angle}_H0max={H0_max}_Ms={Ms_list[1]}_Rmag={R_mag}_gap={gap}.png")
    return Px[-1], Py[-1]

#for GAP in [gap]:
#    for MS in [1]:
#        for R_MAG in [0.2]:
#            alfa_mu_list, Ms_list, rc_list_mag, rc_list_piez = set_geom(angle=180, Ms=MS, R_mag=R_MAG, gap=GAP)
#            calc(alfa_list=alfa_mu_list, Ms_list = Ms_list, rc_list_mag=rc_list_mag, rc_list_piez=rc_list_piez, angle=180, R_mag=R_MAG, gap=GAP)


main_folder = "rot_mag_part"
conf_type="2part"
alf=[i*1+22 for i in range(300)]
Px = []
Py=[]
for i in alf:
    alfa_mu_list, Ms_list, rc_list_mag, rc_list_piez = set_geom_modif(angle=i, conf_type=conf_type)
    px, py = calc(alfa_list=alfa_mu_list, Ms_list = Ms_list, rc_list_mag=rc_list_mag, rc_list_piez=rc_list_piez, angle=i, main_dir=main_folder, conf_type = conf_type)
    Px.append(px)
    Py.append(py)
    ff = open(f"{main_folder}({alf[0]}-{alf[-1]}).txt", "a")
    ff.write(f"{i}\t{px}\t{py}\n")
    ff.close()
plt.figure(clear=True)
plt.plot(alf, Px, label=f"Px(H_max={H0_max})")
plt.plot(alf, Py, label=f"Py(H_max={H0_max})")
plt.xlabel("Угол магн-пьезо-магн, у.е.")
plt.ylabel("Поляризация, у.е.")
plt.legend()
plt.savefig(f"{main_folder}({alf[0]}-{alf[-1]})_el_mag_infl.png")
plt.figure(clear=True)
plt.plot(alf, Px, label=f"Px(H_max={H0_max})")
plt.xlabel("Угол магн-пьезо-магн, у.е.")
plt.ylabel("Поляризация, у.е.")
plt.legend()
plt.savefig(f"{main_folder}({alf[0]}-{alf[-1]})_el_mag_infl_Px.png")
plt.figure(clear=True)
plt.plot(alf, Py, label=f"Py(H_max={H0_max})")
plt.xlabel("Угол магн-пьезо-магн, у.е.")
plt.ylabel("Поляризация, у.е.")
plt.legend()
plt.savefig(f"{main_folder}({alf[0]}-{alf[-1]})_el_mag_infl_Py.png")
plt.figure(clear=True)

t2 = time.time()
print("run time:", t2-t1, "sec")