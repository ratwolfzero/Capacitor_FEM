# add this function
                   
def test_infinite_plates(h=0.5e-3, eps_r=1.0, voltage=100.0):
    """Test case: Infinite parallel plates with uniform dielectric.
    Defaults to eps_r=1.0 (vacuum/air) to match the classic textbook formula:
    C = ε₀ * width / gap
    """
    Lx = 20e-3                    # plates span full domain width
    Ly = 10e-3
    plate_t = 1e-3
    gap = 4e-3
    y0 = 2e-3
    
    bottom = Rectangle(0, y0, Lx, plate_t, voltage=0.0)
    top = Rectangle(0, y0 + plate_t + gap, Lx, plate_t, voltage=voltage)
    dielectric = Rectangle(0, y0 + plate_t, Lx, gap, eps_r=eps_r)
    
    conductors = [bottom, top]
    
    nx = round(Lx / h) + 1
    ny = round(Ly / h) + 1
    mesh = Mesh(0, 0, Lx, Ly, nx=nx, ny=ny)
    
    eps_r_of_xy = make_eps_r_function([dielectric])
    eps_elem = evaluate_material(mesh, eps_r_of_xy)
    
    K, area, area2, b, c = assemble_stiffness(mesh, eps_elem)
    V, is_fixed, solve_time = apply_conductors_and_solve(mesh, K, conductors)
    
    _, _, _, _, _, _, W = compute_fields(mesh, V, eps_elem, b, c, area, area2)
    C_fem = capacitance_from_energy(W, voltage, 0.0)
    
    C_ideal = 8.8541878128e-12 * eps_r * Lx / gap   # exact formula
    
    error_pct = 100 * (C_fem - C_ideal) / C_ideal if C_ideal != 0 else 0.0
    
    print(f"h = {h*1e3:5.3f} mm | nodes = {mesh.n_nodes:6d} | "
          f"C_FEM = {C_fem*1e12:12.6f} pF/m | error = {error_pct:+.6f}% "
          f"(eps_r={eps_r})")
    
    return C_fem, C_ideal, error_pct


#add to main

          
# Run the infinite-plates test ===
    print("\n" + "="*72)
    print("TEST: Infinite parallel plates (full-width, validation case)")
    print("="*72)
    
    
    test_infinite_plates(h=0.5e-3, eps_r=1, voltage=100.0)
    test_infinite_plates(h=0.25e-3, eps_r=1, voltage=100.0)   # finer mesh
    test_infinite_plates(h=0.1e-3,  eps_r=1, voltage=100.0)
