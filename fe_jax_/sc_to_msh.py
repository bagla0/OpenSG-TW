
import numpy as np

def generate_msh_from_sc(input_filepath, output_filepath):
    # 1. Read all non-empty lines from the file
    with open(input_filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # 2. Find the metadata line
    meta_idx = -1
    for i, line in enumerate(lines):
        clean_line = line.split('#')[0]
        parts = clean_line.split()
        if len(parts) >= 5:
            meta_idx = i
            dim = int(parts[0])
            n_nodes = int(parts[1])
            n_elems = int(parts[2])
            n_mats = int(parts[3])
            n_ang = int(parts[-1])
            trans = int(lines[i - 1].split()[2])
            break

    if meta_idx == -1:
        raise ValueError("Could not find the metadata row defining mesh size.")

    # 3. Slice the lists for nodes, elements, and materials
    end_nodes = meta_idx + 1 + n_nodes
    end_elems = end_nodes + n_elems
    node_lines = lines[meta_idx + 1 : end_nodes]
    elem_lines = lines[meta_idx + 1 + n_nodes : end_elems]

    # ---------------------------------------------------------
    # NEW: 3.1 Extract Element Angles
    # ---------------------------------------------------------
    elem_angles = []
    end_elem_angles = end_elems  # Start here by default
    print('p',meta_idx)

    if trans == 1:
        # 1. Locate the block start
        start_ptr = meta_idx + 1 + n_nodes + n_elems
        end_elem_angles = start_ptr + n_elems
        elem_angle_lines = lines[start_ptr : end_elem_angles]

        # 2. Bulk load into a single NumPy array (vectorization starts here)
        # data shape: (n_elems, 7) assuming: ID, e1x, e1y, e1z, e2x, e2y, e2z
        data = np.array([line.split() for line in elem_angle_lines], dtype=float)
        
        e1 = data[:, 1:4]
        e2 = data[:, 4:7]
        
        # 3. Vectorized normalization of e2 (converting to unit vectors)
        # Calculate the norm of every row at once
        norm_e2 = np.linalg.norm(e2, axis=1, keepdims=True)
        # Divide by norm (where norm > 0 to prevent division by zero errors)
        e2 = np.divide(e2, norm_e2, out=np.zeros_like(e2), where=norm_e2 != 0)

        # 4. Vectorized cross product (e3 will have shape (n_elems, 3))
        e3 = np.cross(e1, e2)

        # 5. Stack into the final (N, 9) matrix array
        # This replaces the loop and the multiple .append() calls
        elem_angles = np.hstack([e1, e2, e3])

    # 4. Write the .msh file and track cell types
    cell_types_found = set()    

    with open(output_filepath, 'w') as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        
        # ==========================================
        # 1. NODES SECTION
        # ==========================================
        f.write("$Nodes\n")
        f.write(f"{n_nodes}\n")
        
        # Split logic entirely by dimension to remove inner 'if' checks
        if dim == 1:
            for line in node_lines:
                parts = line.split()
                # Only parse x. Hardcode y and z to 0.0
                f.write(f"{parts[0]} {float(parts[1]):.8f} 0.00000000 0.00000000\n")
                
        elif dim == 2:
            for line in node_lines:
                parts = line.split()
                # Parse x, y. Hardcode z to 0.0
                f.write(f"{parts[0]} {float(parts[1]):.8f} {float(parts[2]):.8f} 0.00000000\n")
                
        elif dim == 3:
            for line in node_lines:
                parts = line.split()
                # Parse x, y, z directly
                f.write(f"{parts[0]} {float(parts[1]):.8f} {float(parts[2]):.8f} {float(parts[3]):.8f}\n")
                
        f.write("$EndNodes\n")

        # ==========================================
        # 2. ELEMENTS SECTION
        # ==========================================
        f.write("$Elements\n")
        f.write(f"{n_elems}\n")
        
        if dim == 1:
            for line in elem_lines:
                parts = line.split()
                elem_id, mat_id = parts[0], parts[1] 
                
                connectivity = [n for n in parts[2:] if n != '0']
                num_nodes = len(connectivity)
                
                if num_nodes == 2: 
                    elem_type = 1
                    cell_types_found.add("2-Node Interval")
                elif num_nodes == 3: 
                    elem_type = 8
                    cell_types_found.add("3-Node Interval")
                elif num_nodes == 4: 
                    elem_type = 26
                    cell_types_found.add("4-Node Interval")
                elif num_nodes == 5: 
                    elem_type = 27  # Correctly maps to 27
                    cell_types_found.add("5-Node Interval")
                else: elem_type = 1 
                    
                f.write(f"{elem_id} {elem_type} 2 {mat_id} {mat_id} {' '.join(connectivity)}\n")

        elif dim == 2:
            for line in elem_lines:
                parts = line.split()
                elem_id, mat_id = parts[0], parts[1]
                
                # Filter out trailing zeros used for padding degenerated elements
                connectivity = [n for n in parts[2:] if n != '0']
                num_nodes = len(connectivity)
                
                # Only Triangles and Quadrilaterals allowed in 2D
                if num_nodes == 3:
                    elem_type = 2
                    cell_types_found.add("3-Node Triangle")
                elif num_nodes == 4:
                    elem_type = 3
                    cell_types_found.add("4-Node Quadrilateral")
                else:
                    elem_type = 2 # Default fallback
                    cell_types_found.add(f"Unknown 2D Type ({num_nodes} nodes)")
                    
                f.write(f"{elem_id} {elem_type} 2 {mat_id} {mat_id} {' '.join(connectivity)}\n")

        elif dim == 3:
            for line in elem_lines:
                parts = line.split()
                elem_id, mat_id = parts[0], parts[1]
                
                # Get the raw list of nodes before filtering anything
                raw_connectivity = parts[2:]
                
                # Check the .sc convention: If the 5th node (index 4) is '0', it's a Tetra
                # We use len() >= 5 to prevent index errors on shorter lines
                is_tetra_deg2 = raw_connectivity[5] != '0'
                
                if is_tetra_deg2:
                    connectivity = raw_connectivity[:4]  + raw_connectivity[6:12]
                    elem_type = 11
                    cell_types_found.add("10-Node Tetrahedron")  
                    
                else:
                    elem_type = 4
                    connectivity = raw_connectivity[:4]
                    cell_types_found.add("4-Node Tetrahedron")


               
                f.write(f"{elem_id} {elem_type} 2 {mat_id} {mat_id} {' '.join(connectivity)}\n")
                
        f.write("$EndElements\n")

    # ==========================================
    # 6. PRINT SUMMARY
    # ==========================================
    print("--- MESH SUMMARY ---\n")
    print(f"SG:          {dim}D")
    print(f"Num Nodes:   {n_nodes}")
    print(f"Num Elem:   {n_elems}")
    print(f"Num Mat:     {n_mats}")
    print(f"Cell Types:  {', '.join(cell_types_found)}")

        # ---------------------------------------------------------
    # NEW: 3.2 Extract Material Angles of cell ID
    # ---------------------------------------------------------
 
    end_mat_angles = end_elem_angles + n_ang
    mat_angle_lines = lines[end_elem_angles : end_mat_angles]
    
    mat_angles = []
    mat_seq = []
    for line in mat_angle_lines:
        parts = line.split()
        mat_angles.append(float(parts[2]))
        mat_seq.append(int(parts[1])-1)
    # ---------------------------------------------------------
    # NEW: 3.3 Extract Material Parameters
    # ---------------------------------------------------------
    mat_params = []
    for i in range(n_mats):

        start = end_mat_angles + (i * 5)
        props = []
        for line in lines[start + 2 : start + 5]:
            props.extend((float(val) for val in line.split()))
            
        mat_params.append(props)

    print(f"Num Angles:  {n_ang}")

    return dim, elem_angles, mat_angles, mat_seq, mat_params