"""
结果分析模块
负责 MD 模拟结束后的数据解析和相互作用分析。
包含：
  - parse_xvg: 解析 GROMACS .xvg 输出文件
  - analyze_interactions: 基于 PDB 的蛋白质-配体相互作用分析（氢键、疏水接触）
"""

import os

# 氨基酸 3 字母→1 字母映射
AA_THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

# 疏水残基（非极性）
HYDROPHOBIC_AAS = {'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'MET', 'PRO', 'TRP', 'GLY'}

# 氢键供体原子
DONOR_ATOMS = {'N', 'O', 'OG', 'OG1', 'OD1', 'OD2', 'OE1', 'OE2',
               'ND1', 'ND2', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'OH', 'SG'}
# 氢键受体原子
ACCEPTOR_ATOMS = {'O', 'OD1', 'OD2', 'OE1', 'OE2', 'ND1', 'NE2',
                  'OG', 'OG1', 'OH', 'N', 'NZ'}


def parse_xvg(filepath, columns=1, skip_rows=0):
    """
    解析 .xvg 文件。
    .xvg 文件以 @ 或 # 开头的是注释/元数据行，数据行是纯数字。
    columns: 返回的列索引
    skip_rows: 跳过的初始行数
    """
    data = {'time': []}
    for col_idx in range(columns):
        data[f'col_{col_idx}'] = []

    try:
        with open(filepath) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return data

    row_count = 0
    for line in lines:
        line = line.strip()
        if not line or line.startswith('@') or line.startswith('#'):
            continue

        parts = line.split()
        if len(parts) < columns + 1:
            continue

        row_count += 1
        if row_count <= skip_rows:
            continue

        try:
            data['time'].append(float(parts[0]))
            for col_idx in range(columns):
                data[f'col_{col_idx}'].append(float(parts[col_idx + 1]))
        except (ValueError, IndexError):
            continue

    return data


def _distance(a, b):
    """计算两个原子之间的欧几里得距离（Å）"""
    return ((a['x'] - b['x']) ** 2 +
            (a['y'] - b['y']) ** 2 +
            (a['z'] - b['z']) ** 2) ** 0.5


def _resname_to_one(three):
    """将三位氨基酸代码转换为一字母代码"""
    return AA_THREE_TO_ONE.get(three, 'X')


def analyze_interactions(pdb_path, log_func=None):
    """
    基于原始 PDB 文件，用几何方法分析蛋白-配体相互作用。

    参数:
        pdb_path: PDB 文件路径
        log_func: 可选的日志回调函数，接受 (message, level) 参数

    返回:
        dict: {
            'has_ligand': bool,
            'hbonds': {'count': int, 'details': [...], 'per_residue': [...]},
            'hydrophobic': {'count': int, 'details': [...], 'per_residue': [...]},
        }
        如果找不到配体，返回 has_ligand=False 的默认值。
    """
    def log(msg, level='info'):
        if log_func:
            log_func(msg, level)

    if not os.path.exists(pdb_path):
        log('未找到原始 PDB 文件，跳过相互作用分析。', 'warning')
        return _default_result()

    # ---------- 解析 PDB ----------
    protein_atoms = []
    ligand_atoms = []

    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith('ATOM'):
                    atomname = line[12:16].strip()
                    resname = line[17:20].strip()
                    resid = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    element = atomname[0] if atomname else 'X'
                    protein_atoms.append({
                        'x': x, 'y': y, 'z': z,
                        'resname': resname,
                        'resid': resid,
                        'atomname': atomname,
                        'element': element,
                    })
                elif line.startswith('HETATM'):
                    atomname = line[12:16].strip()
                    resname = line[17:20].strip()
                    resid = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    element = line[76:78].strip() or atomname[0]
                    if resname in ('HOH', 'SOL', 'WAT', 'NA', 'CL', 'K', 'MG', 'CA'):
                        continue
                    ligand_atoms.append({
                        'x': x, 'y': y, 'z': z,
                        'atomname': atomname,
                        'element': element,
                        'resname': resname,
                        'resid': resid,
                    })
    except Exception as e:
        log(f'PDB 解析失败: {e}', 'error')
        return _default_result()

    if not ligand_atoms:
        log('未检测到配体（HETATM 记录），跳过相互作用分析。', 'info')
        return _default_result()

    if not protein_atoms:
        return _default_result()

    # ---------- 1. 氢键分析 ----------
    hbonds = []
    for lig in ligand_atoms:
        for prot in protein_atoms:
            d = _distance(lig, prot)
            if d < 3.5:
                lig_is_donor = lig['element'] in ('N', 'O')
                prot_is_acceptor = (prot['atomname'] in ACCEPTOR_ATOMS or
                                    prot['element'] in ('O', 'N'))
                prot_is_donor = (prot['atomname'] in DONOR_ATOMS or
                                 prot['element'] in ('N', 'O'))
                if lig_is_donor and prot_is_acceptor:
                    hbonds.append({
                        'ligand_atom': lig['atomname'],
                        'protein_residue': f"{prot['resname']}{prot['resid']}",
                        'protein_atom': prot['atomname'],
                        'distance_angstrom': round(d, 2),
                        'role': '配体供体',
                    })
                elif prot_is_donor and (lig['element'] in ('O', 'N')):
                    hbonds.append({
                        'ligand_atom': lig['atomname'],
                        'protein_residue': f"{prot['resname']}{prot['resid']}",
                        'protein_atom': prot['atomname'],
                        'distance_angstrom': round(d, 2),
                        'role': '蛋白供体',
                    })

    # 去重：同一对残基-配体原子只保留最近的
    seen = set()
    unique_hbonds = []
    for h in sorted(hbonds, key=lambda x: x['distance_angstrom']):
        key = (h['protein_residue'], h['ligand_atom'])
        if key not in seen:
            seen.add(key)
            unique_hbonds.append(h)
    hbonds = unique_hbonds

    # 按残基统计氢键数量
    hbond_counts = {}
    for h in hbonds:
        res = h['protein_residue']
        hbond_counts[res] = hbond_counts.get(res, 0) + 1
    hbond_residues_sorted = sorted(hbond_counts.items(), key=lambda x: -x[1])

    # ---------- 2. 疏水接触分析 ----------
    hydrophobic_contacts = []
    for lig in ligand_atoms:
        if lig['element'] not in ('C',):
            continue
        for prot in protein_atoms:
            if prot['resname'] not in HYDROPHOBIC_AAS:
                continue
            if prot['element'] not in ('C',):
                continue
            d = _distance(lig, prot)
            if d < 4.0:
                hydrophobic_contacts.append({
                    'ligand_atom': lig['atomname'],
                    'protein_residue': f"{prot['resname']}{prot['resid']}",
                    'distance_angstrom': round(d, 2),
                })

    # 去重：每对残基-配体原子只保留一次
    seen_hydro = set()
    unique_hydro = []
    for h in sorted(hydrophobic_contacts, key=lambda x: x['distance_angstrom']):
        key = (h['protein_residue'], h['ligand_atom'])
        if key not in seen_hydro:
            seen_hydro.add(key)
            unique_hydro.append(h)
    hydrophobic_contacts = unique_hydro

    # 按残基统计疏水接触
    hydro_counts = {}
    for h in hydrophobic_contacts:
        res = h['protein_residue']
        hydro_counts[res] = hydro_counts.get(res, 0) + 1
    hydro_residues_sorted = sorted(hydro_counts.items(), key=lambda x: -x[1])

    # ---------- 整理结果 ----------
    result = {
        'has_ligand': True,
        'hbonds': {
            'count': len(hbonds),
            'details': hbonds,
            'per_residue': [
                {
                    'residue': r,
                    'count': c,
                    'label': f'{_resname_to_one(r.rstrip("0123456789"))}{r.lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}'
                }
                for r, c in hbond_residues_sorted
            ],
        },
        'hydrophobic': {
            'count': len(hydrophobic_contacts),
            'details': hydrophobic_contacts,
            'per_residue': [{'residue': r, 'count': c} for r, c in hydro_residues_sorted],
        },
    }

    log(f'检测到 {len(hbonds)} 个氢键、{len(hydrophobic_contacts)} 个疏水接触', 'info')
    return result


def _default_result():
    """返回无配体时的默认结果"""
    return {
        'has_ligand': False,
        'hbonds': {'count': 0, 'details': [], 'per_residue': []},
        'hydrophobic': {'count': 0, 'details': [], 'per_residue': []},
    }
