from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
import json

def analyze_peptide(peptide_sequence: str):
    # アミノ酸配列から分子を生成（例: "GG" など、大文字1文字表記）
    mol = Chem.MolFromSequence(peptide_sequence)
    if not mol:
        raise ValueError("無効なアミノ酸配列です。")
        
    # 水素を付加し、初期3D座標を生成
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol) # 簡易構造最適化
    
    conformer = mol.GetConformer()
    
    # 1. 各原子の親疎水性（Crippen LogP寄与度）の計算
    # (原子ごとのLogP寄与, MR寄与) のタプルリストが返る
    crippen_contribs = rdMolDescriptors._CalcCrippenContribs(mol)
    
    # 2. 水素結合ドナー・アクセプターの定義（SMARTSパターンで判定）
    # 簡易的な判定基準
    donor_query = Chem.MolFromSmarts("[$[N,O;H1,H2,H3]]")
    acceptor_query = Chem.MolFromSmarts("[$[O,N;H0,H1,H2,H3]]")
    
    donors = set(mol.GetSubstructMatches(donor_query))
    acceptors = set(mol.GetSubstructMatches(acceptor_query))
    
    # 原子リストの構築
    atoms_data = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conformer.GetAtomPosition(idx)
        
        # 元素記号
        element = atom.GetSymbol()
        
        # 親疎水性 (LogP)
        logp = crippen_contribs[idx][0]
        
        # 水素結合属性
        h_bond = "none"
        if (idx,) in donors:
            h_bond = "donor"
        elif (idx,) in acceptors:
            h_bond = "acceptor"
            
        # van der Waals半径 (簡易マッピング)
        vdw_radii = {"H": 1.2, "C": 1.7, "N": 1.55, "O": 1.5, "S": 1.8}
        radius = vdw_radii.get(element, 1.5)
        
        atoms_data.append({
            "id": idx,
            "element": element,
            "pos": [pos.x, pos.y, pos.z],
            "hydrophobicity": logp,
            "h_bond": h_bond,
            "radius": radius
        })
        
    # 3. 可動軸と結合トポロジーの抽出
    bonds_data = []
    for bond in mol.GetBonds():
        source = bond.GetBeginAtomIdx()
        target = bond.GetEndAtomIdx()
        
        # アミド結合（ペプチド結合）の簡易判定
        # C-N 結合かつ、Cがカルボニル(Oとダブルボンド)である場合
        is_amide = False
        if bond.GetBondType() == Chem.BondType.SINGLE:
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            if (begin_atom.GetSymbol() == 'C' and end_atom.GetSymbol() == 'N') or \
               (begin_atom.GetSymbol() == 'N' and end_atom.GetSymbol() == 'C'):
                # 隣接する酸素に二重結合があるかチェック
                c_atom = begin_atom if begin_atom.GetSymbol() == 'C' else end_atom
                for nb in c_atom.GetNeighbors():
                    if nb.GetSymbol() == 'O':
                        nb_bond = mol.GetBondBetweenAtoms(c_atom.GetIdx(), nb.GetIdx())
                        if nb_bond.GetBondType() == Chem.BondType.DOUBLE:
                            is_amide = True
                            break
                            
        # 初期結合長
        pos_s = conformer.GetAtomPosition(source)
        pos_t = conformer.GetAtomPosition(target)
        dist = ((pos_s.x - pos_t.x)**2 + (pos_s.y - pos_t.y)**2 + (pos_s.z - pos_t.z)**2)**0.5
        
        bonds_data.append({
            "source": source,
            "target": target,
            "length": dist,
            "is_amide": is_amide
        })
        
    payload = {
        "atoms": atoms_data,
        "bonds": bonds_data
    }
    return json.dumps(payload, indent=2)

# テスト実行（アラニルグリシンなどの短いペプチドでテスト可能）
# print(analyze_peptide("AG"))