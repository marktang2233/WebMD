"""
MDP 参数模板
支持加水/真空两种模式：
  - solvate=True: 使用 Protein SOL 能量组、Protein Non-Protein 温度组（标准）
  - solvate=False: 使用 Protein 能量组、System 温度组（真空）
"""

def em_mdp(solvate=True):
    energygrps = 'Protein SOL' if solvate else 'Protein'
    dispcorr = '\nDispCorr                 = EnerPres' if solvate else ''
    coulomb = 'PME\nrcoulomb                 = 1.2' if solvate else 'Cut-off\nrcoulomb                 = 2.0'
    rl = 'rlist                    = 1.2' if solvate else 'rlist                    = 2.0'
    rvdw_val = '1.2' if solvate else '2.0'
    return f"""
; Energy Minimization (最陡下降法能量最小化)
define                   = -DFLEXIBLE
integrator               = steep
nsteps                   = 50000
nstenergy                = 10
nstlog                   = 10
energygrps               = {energygrps}
cutoff-scheme            = Verlet
ns-type                  = grid
nstlist                  = 10
{rl}
coulombtype              = {coulomb}
vdwtype                  = Cut-off
rvdw                     = {rvdw_val}{dispcorr}
pbc                      = xyz
"""


def nvt_mdp(ref_temp, solvate=True):
    energygrps = 'Protein SOL' if solvate else 'Protein'
    tc_grps = 'Protein Non-Protein' if solvate else 'System'
    tau_t = '0.1   0.1' if solvate else '0.1'
    ref_t = f'{ref_temp}   {ref_temp}' if solvate else f'{ref_temp}'
    coulomb = 'PME\nrcoulomb                 = 1.2' if solvate else 'Cut-off\nrcoulomb                 = 2.0'
    rl = 'rlist                    = 1.2' if solvate else 'rlist                    = 2.0'
    rvdw_val = '1.2' if solvate else '2.0'
    return f"""
; NVT Equilibration (等温等容平衡)
define                   = -DPOSRES
integrator               = md
nsteps                   = 50000
dt                       = 0.002
nstxout                  = 500
nstvout                  = 500
nstenergy                = 100
nstlog                   = 100
nstxout-compressed       = 500
compressed-x-grps        = System
energygrps               = {energygrps}
cutoff-scheme            = Verlet
ns-type                  = grid
nstlist                  = 10
{rl}
coulombtype              = {coulomb}
vdwtype                  = Cut-off
rvdw                     = {rvdw_val}
pbc                      = xyz
tcoupl                   = V-rescale
tc-grps                  = {tc_grps}
tau_t                    = {tau_t}
ref_t                    = {ref_t}
pcoupl                   = no
gen-vel                  = yes
gen-temp                 = {ref_temp}
gen-seed                 = -1
constraints              = h-bonds
constraint-algorithm     = LINCS
continuation             = no
"""


def npt_mdp(ref_temp, solvate=True):
    energygrps = 'Protein SOL' if solvate else 'Protein'
    tc_grps = 'Protein Non-Protein' if solvate else 'System'
    tau_t = '0.1   0.1' if solvate else '0.1'
    ref_t = f'{ref_temp}   {ref_temp}' if solvate else f'{ref_temp}'
    coulomb = 'PME\nrcoulomb                 = 1.2' if solvate else 'Cut-off\nrcoulomb                 = 2.0'
    rl = 'rlist                    = 1.2' if solvate else 'rlist                    = 2.0'
    rvdw_val = '1.2' if solvate else '2.0'
    # 真空模式下不加压（无溶剂）
    pcoupl_section = '\npcoupl                   = Berendsen\npcoupltype               = isotropic\ntau_p                    = 2.0\nref_p                    = 1.0\ncompressibility          = 4.5e-5' if solvate else '\npcoupl                   = no'
    return f"""
; NPT Equilibration (等温等压平衡)
define                   = -DPOSRES
integrator               = md
nsteps                   = 50000
dt                       = 0.002
nstxout                  = 500
nstvout                  = 500
nstenergy                = 100
nstlog                   = 100
nstxout-compressed       = 500
compressed-x-grps        = System
energygrps               = {energygrps}
cutoff-scheme            = Verlet
ns-type                  = grid
nstlist                  = 10
{rl}
coulombtype              = {coulomb}
vdwtype                  = Cut-off
rvdw                     = {rvdw_val}
pbc                      = xyz
tcoupl                   = V-rescale
tc-grps                  = {tc_grps}
tau_t                    = {tau_t}
ref_t                    = {ref_t}{pcoupl_section}
gen-vel                  = no
constraints              = h-bonds
constraint-algorithm     = LINCS
continuation             = yes
"""


def md_mdp(nsteps, ref_temp, solvate=True):
    energygrps = 'Protein SOL' if solvate else 'Protein'
    tc_grps = 'Protein Non-Protein' if solvate else 'System'
    tau_t = '0.1   0.1' if solvate else '0.1'
    ref_t = f'{ref_temp}   {ref_temp}' if solvate else f'{ref_temp}'
    coulomb = 'PME\nrcoulomb                 = 1.2' if solvate else 'Cut-off\nrcoulomb                 = 2.0'
    rl = 'rlist                    = 1.2' if solvate else 'rlist                    = 2.0'
    rvdw_val = '1.2' if solvate else '2.0'
    pcoupl_section = '\npcoupl                   = Parrinello-Rahman\npcoupltype               = isotropic\ntau_p                    = 2.0\nref_p                    = 1.0\ncompressibility          = 4.5e-5' if solvate else '\npcoupl                   = no'
    return f"""
; Production MD (生产模拟，收集数据)
integrator               = md
nsteps                   = {nsteps}
dt                       = 0.002
nstxout                  = 0
nstvout                  = 0
nstenergy                = 100
nstlog                   = 100
nstxout-compressed       = 500
compressed-x-grps        = System
energygrps               = {energygrps}
cutoff-scheme            = Verlet
ns-type                  = grid
nstlist                  = 10
{rl}
coulombtype              = {coulomb}
vdwtype                  = Cut-off
rvdw                     = {rvdw_val}
pbc                      = xyz
tcoupl                   = V-rescale
tc-grps                  = {tc_grps}
tau_t                    = {tau_t}
ref_t                    = {ref_t}{pcoupl_section}
gen-vel                  = no
constraints              = h-bonds
constraint-algorithm     = LINCS
continuation             = yes
"""


# 保留旧常量以兼容旧代码（如果需要）
IONS_MDP = """
; ions.mdp - used to generate ions.tpr with grompp
integrator               = steep
nsteps                   = 1
"""
