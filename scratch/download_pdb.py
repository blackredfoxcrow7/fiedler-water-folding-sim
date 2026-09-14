import urllib.request
import os
import numpy as np

def download_pdb(pdb_id):
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    filename = f"{pdb_id}.pdb"
    if not os.path.exists(filename):
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, filename)
    else:
        print(f"{filename} already exists.")
    return filename

def parse_heavy_atoms_pdb(filename):
    # Parse heavy atoms for Model 1
    coords = []
    atoms_info = []
    
    with open(filename, 'r') as f:
        in_model1 = False
        has_models = False
        
        # Check if PDB has MODEL records
        for line in f:
            if line.startswith("MODEL"):
                has_models = True
                break
        
        f.seek(0)
        
        for line in f:
            if has_models:
                if line.startswith("MODEL        1"):
                    in_model1 = True
                    continue
                elif line.startswith("ENDMDL"):
                    if in_model1:
                        break
            else:
                in_model1 = True
                
            if in_model1 and (line.startswith("ATOM") or line.startswith("HETATM")):
                element = line[76:78].strip()
                if not element:
                    # Fallback to atom name first char if element column is empty
                    atom_name = line[12:16].strip()
                    element = atom_name[0]
                
                # We only want heavy atoms (no Hydrogen)
                if element == 'H':
                    continue
                    
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                
                res_num = int(line[22:26])
                res_name = line[17:20].strip()
                atom_name = line[12:16].strip()
                
                coords.append([x, y, z])
                atoms_info.append({
                    "element": element,
                    "res_num": res_num,
                    "res_name": res_name,
                    "atom_name": atom_name
                })
                
    return np.array(coords), atoms_info

if __name__ == "__main__":
    import numpy as np
    filename = download_pdb("1UAO")
    coords, info = parse_heavy_atoms_pdb(filename)
    print(f"Loaded {len(coords)} heavy atoms from model 1 of 1UAO.")
    for i in range(min(15, len(info))):
        print(f"  Atom {i}: {info[i]['res_name']}{info[i]['res_num']} {info[i]['atom_name']} ({info[i]['element']}) -> {coords[i]}")
