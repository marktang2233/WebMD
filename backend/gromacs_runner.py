import os
import re
import json
import time
import uuid
import shutil
import threading
import subprocess
from datetime import datetime

from mdp_templates import IONS_MDP, em_mdp, nvt_mdp, npt_mdp, md_mdp
from analysis import parse_xvg, analyze_interactions

class GromacsRunner:
    """
    管理所有 GROMACS 模拟任务的状态和执行。
    每个任务在独立线程中运行，支持暂停/恢复/取消。
    """

    def __init__(self, tasks_dir, gmx_command='gmx'):
        self.tasks_dir = tasks_dir
        self.gmx = gmx_command  # GROMACS 可执行文件路径
        self.tasks = {}         # 所有任务状态 dict，key=task_id

        if not os.path.exists(tasks_dir):
            os.makedirs(tasks_dir)

        self._tasks_db_path = os.path.join(self.tasks_dir, '_tasks_db.json')
        self._load_tasks()

    def _load_tasks(self):
        """从 JSON 文件恢复任务状态，并从 tasks 目录中补充丢失的任务"""
        self.tasks = {}
        if os.path.exists(self._tasks_db_path):
            try:
                with open(self._tasks_db_path, 'r') as f:
                    data = json.load(f)
                for task in data.values():
                    if task.get('status') in ('running', 'pending', 'paused'):
                        task['status'] = 'failed'
                        task['completed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        if 'logs' not in task:
                            task['logs'] = []
                        task['logs'].append({
                            'time': datetime.now().strftime('%H:%M:%S'),
                            'level': 'error',
                            'message': '服务重启，该任务已终止。'
                        })
                self.tasks = data
                print(f"从数据库加载了 {len(self.tasks)} 个任务")
            except Exception as e:
                print(f"加载持久化任务失败: {e}")
                self.tasks = {}

        try:
            self._recover_missing_tasks()
        except Exception as e:
            print(f"任务恢复过程出错: {e}")
            import traceback
            traceback.print_exc()

        # 为所有已加载的任务补充缺失的步骤时间戳
        self._fix_loaded_tasks_step_timing()

    def _recover_missing_tasks(self):
        """扫描 tasks 目录，恢复数据库文件中丢失的任务"""
        recovered = 0
        if not os.path.exists(self.tasks_dir):
            print(f"tasks 目录不存在: {self.tasks_dir}")
            return

        entries = [e for e in os.listdir(self.tasks_dir)]
        print(f"扫描 tasks 目录，发现 {len(entries)} 个条目")
        for entry in entries:
            task_dir = os.path.join(self.tasks_dir, entry)
            if not os.path.isdir(task_dir):
                continue
            if entry.startswith('_') or entry.startswith('.'):
                continue
            if entry in self.tasks:
                continue

            params_path = os.path.join(task_dir, 'params.json')
            if not os.path.exists(params_path):
                print(f"  跳过 {entry}: 无 params.json")
                continue

            try:
                with open(params_path, 'r') as f:
                    params = json.load(f)
            except Exception as e:
                print(f"  跳过 {entry}: params.json 读取失败: {e}")
                continue

            task_type = 'md'
            task_title = (params.get('description', '') or params.get('pdb_filename', '')
                         or params.get('molecule_filename', '') or entry)

            steps = []
            is_gaussian = bool(entry.startswith('gauss_') or params.get('source_task_id'))
            if is_gaussian:
                task_type = 'gaussian'
                steps.append({'id': 'gaussian', 'name': '量子化学计算 (Gaussian)',
                             'description': 'Gaussian 量子化学计算', 'status': 'pending', 'detail': ''})
            else:
                if params.get('gaussian_enabled') and params.get('ligand_path'):
                    steps.append({'id': 'gaussian', 'name': '量子化学计算 (Gaussian)',
                                 'description': 'Gaussian 配体量子化学计算', 'status': 'pending', 'detail': ''})
                if params.get('docking_enabled') and params.get('ligand_path'):
                    steps.append({'id': 'docking', 'name': '分子对接',
                                 'description': 'AutoDock Vina 蛋白-配体对接', 'status': 'pending', 'detail': ''})
                for sid, sname, sdesc in [
                    ('em',  '能量最小化 (EM)', '最陡下降最小化以消除空间冲突。'),
                    ('nvt', 'NVT 平衡',         '恒定粒子数、体积、温度。'),
                    ('npt', 'NPT 平衡',         '恒定粒子数、压力、温度。'),
                    ('md',  '生产模拟',         '生产运行以收集数据。'),
                ]:
                    steps.append({'id': sid, 'name': sname, 'description': sdesc,
                                 'status': 'pending', 'detail': ''})

            status, progress = self._infer_task_status_and_progress(task_dir, steps, is_gaussian)

            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            completed_at = now_str if status == 'completed' else None
            results_ready = status == 'completed'

            self.tasks[entry] = {
                'task_id': entry,
                'params': params,
                'task_type': task_type,
                'title': task_title,
                'description': params.get('description', ''),
                'submitted_at': self._guess_submitted_at(task_dir, entry),
                'status': status,
                'progress': progress,
                'paused': False,
                'cancelled': False,
                'current_step': '',
                'logs': [],
                'steps': steps,
                'completed_at': completed_at,
                'results_ready': results_ready,
                'work_dir': task_dir,
            }

            # 恢复 docking_results（如果 docked_ligand.pdbqt 存在）
            docking_pdbqt_path = os.path.join(task_dir, 'docked_ligand.pdbqt')
            if os.path.exists(docking_pdbqt_path) and os.path.getsize(docking_pdbqt_path) > 0:
                try:
                    vina_results = self._parse_vina_output(entry, docking_pdbqt_path)
                    if vina_results:
                        best_mode = vina_results[0]
                        self.tasks[entry]['docking_results'] = {
                            'modes': vina_results,
                            'best_affinity': best_mode['affinity'],
                            'best_rmsd_lb': best_mode['rmsd_lb'],
                            'best_rmsd_ub': best_mode['rmsd_ub'],
                        }
                except Exception as e:
                    print(f"  恢复 {entry} docking_results 失败: {e}")

            # 恢复 gaussian_results（如果 ligand.log 存在）
            gaussian_log_path = os.path.join(task_dir, 'ligand.log')
            if os.path.exists(gaussian_log_path):
                try:
                    gaussian_results = self._parse_gaussian_output(entry, gaussian_log_path)
                    if gaussian_results:
                        self.tasks[entry]['gaussian_results'] = gaussian_results
                except Exception as e:
                    print(f"  恢复 {entry} gaussian_results 失败: {e}")

            recovered += 1

        print(f"恢复结果: {recovered} 个任务恢复成功")
        if recovered > 0:
            self._save_tasks()

    def _fix_loaded_tasks_step_timing(self):
        """为已加载任务中缺失时间戳的步骤补充时间信息"""
        import datetime as dt_module
        now_str = dt_module.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        step_file_map = [
            ('gaussian', 'ligand.log'),
            ('docking', 'docked_ligand.pdbqt'),
            ('em', 'em.gro'),
            ('nvt', 'nvt.gro'),
            ('npt', 'npt.gro'),
            ('md', 'md.gro'),
        ]
        for task_id, task in self.tasks.items():
            work_dir = task.get('work_dir', '')
            if not work_dir or not os.path.exists(work_dir):
                continue
            steps = task.get('steps', [])
            task_started = task.get('started_at', '')
            task_completed = task.get('completed_at', now_str)

            # 如果没有 started_at，用 submitted_at 或第一个文件的 mtime
            if not task_started:
                task_started = task.get('submitted_at', '')
                if task_started:
                    task['started_at'] = task_started

            # 按顺序为每个步骤填充缺失的时间戳
            prev_completed = task_started  # 链式起点：任务提交时间
            for step in steps:
                if step.get('started_at') and step.get('completed_at'):
                    prev_completed = step['completed_at']
                    continue

                sid = step['id']

                # 非 completed 状态：只设开始时间，不设完成时间
                if step['status'] != 'completed':
                    if not step.get('started_at') and task_started:
                        step['started_at'] = task_started
                    # 不更新链式指针（非 completed 步骤的时间不可靠）
                    continue

                # completed 步骤：从输出文件的 mtime 获取完成时间
                matched_file = None
                for step_id, filename in step_file_map:
                    if step_id == sid:
                        matched_file = filename
                        break

                completion_ts = ''
                if matched_file:
                    fpath = os.path.join(work_dir, matched_file)
                    if os.path.exists(fpath):
                        mtime = os.path.getmtime(fpath)
                        completion_ts = dt_module.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')

                if not completion_ts:
                    completion_ts = task_completed

                # 设置 started_at = 前一步完成时间（链式）
                if not step.get('started_at'):
                    step['started_at'] = prev_completed if prev_completed else completion_ts
                # 设置 completed_at = 当前文件的 mtime
                if not step.get('completed_at'):
                    step['completed_at'] = completion_ts

                prev_completed = step['completed_at']

    def _infer_task_status_and_progress(self, task_dir, steps, is_gaussian=False):
        """根据工作目录中的输出文件推断任务状态和进度"""
        if is_gaussian:
            if os.path.exists(os.path.join(task_dir, 'ligand.log')):
                for s in steps:
                    if s['id'] == 'gaussian':
                        s['status'] = 'completed'
                        s['detail'] = '已完成'
                return 'completed', 100
            return 'failed', 50

        status_files = ['em.gro', 'nvt.gro', 'npt.gro', 'md.gro']
        step_ids = ['em', 'nvt', 'npt', 'md']

        last_idx = -1
        for i, filename in enumerate(status_files):
            if os.path.exists(os.path.join(task_dir, filename)):
                last_idx = i
            else:
                break

        # 检查是否有 docked_ligand.pdbqt（Vina 实际输出文件名）
        docking_pdbqt = os.path.join(task_dir, 'docked_ligand.pdbqt')
        if os.path.exists(docking_pdbqt) and os.path.getsize(docking_pdbqt) > 0:
            for s in steps:
                if s['id'] == 'docking':
                    s['status'] = 'completed'
                    s['detail'] = '已完成'

        # 检查 Gaussian 输出
        if os.path.exists(os.path.join(task_dir, 'ligand.log')):
            for s in steps:
                if s['id'] == 'gaussian':
                    s['status'] = 'completed'
                    s['detail'] = '已完成'

        if last_idx == -1:
            # 检查 docking/gaussian 是否已完成（独立运行的情况）
            has_docking = any(s['id'] == 'docking' for s in steps)
            has_gaussian = any(s['id'] == 'gaussian' for s in steps)
            if not has_docking and not has_gaussian:
                if os.path.exists(os.path.join(task_dir, 'params.json')):
                    return 'pending', 0
            return 'failed', 50

        for i, sid in enumerate(step_ids):
            if i <= last_idx:
                for s in steps:
                    if s['id'] == sid:
                        s['status'] = 'completed'
                        s['detail'] = '已完成'
                        break
            else:
                for s in steps:
                    if s['id'] == sid and s['status'] == 'pending':
                        s['status'] = 'failed'
                        s['detail'] = '未完成'
                        break

        # MD 已完成，但 docking/gaussian 仍为 pending → 标记为 failed
        if last_idx == len(status_files) - 1:
            has_docked_file = os.path.exists(os.path.join(task_dir, 'docked_ligand.pdbqt'))
            for s in steps:
                if s['id'] == 'docking' and s['status'] == 'pending':
                    if has_docked_file:
                        s['status'] = 'completed'
                        s['detail'] = '已完成'
                    else:
                        s['status'] = 'failed'
                        s['detail'] = '无对接输出文件'
                if s['id'] == 'gaussian' and s['status'] == 'pending':
                    s['status'] = 'failed'
                    s['detail'] = '无 Gaussian 输出文件'
            return 'completed', 100

        # 部分完成 → 标记为 failed（缺少一些 gro 文件）
        # 同时标记未完成的 docking/gaussian 步骤
        for s in steps:
            if s['status'] == 'pending':
                if s['id'] in ('docking', 'gaussian'):
                    s['status'] = 'failed'
                    s['detail'] = f'MD 模拟未完成，{s["name"]}未执行'
        return 'failed', 50

    def _guess_submitted_at(self, task_dir, task_id):
        """从目录修改时间或 task_id 中的时间戳推断提交时间"""
        import datetime as dt_module
        match = re.search(r'(\d{8})_(\d{6})', task_id)
        if match:
            try:
                return dt_module.datetime.strptime(f"{match.group(1)} {match.group(2)}",
                                                    '%Y%m%d %H%M%S').strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                pass
        try:
            mtime = os.path.getmtime(task_dir)
            return dt_module.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def _save_tasks(self):
        """将任务状态保存到 JSON 文件"""
        try:
            with open(self._tasks_db_path, 'w') as f:
                json.dump(self.tasks, f, indent=2, default=str)
        except Exception as e:
            print(f"保存任务状态失败: {e}")

    def delete_task(self, task_id):
        """删除任务：移除内存记录、删除工作目录、更新持久化"""
        if task_id not in self.tasks:
            return False, '任务不存在'

        work_dir = self.tasks[task_id].get('work_dir', '')
        del self.tasks[task_id]
        self._save_tasks()

        if work_dir and os.path.exists(work_dir):
            try:
                shutil.rmtree(work_dir)
            except Exception as e:
                print(f"删除工作目录失败 {work_dir}: {e}")

        return True, '已删除'

    def create_task(self, task_id, params, task_type='md'):
        """创建新任务并初始化状态"""
        steps = []
        if task_type == 'gaussian':
            steps.append({'id': 'gaussian', 'name': '量子化学计算 (Gaussian)', 'description': 'Gaussian 量子化学计算', 'status': 'pending'})
        else:
            if params.get('gaussian_enabled') and params.get('ligand_path'):
                steps.append({'id': 'gaussian', 'name': '量子化学计算 (Gaussian)', 'description': 'Gaussian 配体量子化学计算', 'status': 'pending'})
            if params.get('docking_enabled') and params.get('ligand_path'):
                steps.append({'id': 'docking', 'name': '分子对接', 'description': 'AutoDock Vina 蛋白-配体对接', 'status': 'pending'})
            steps.extend([
                {'id': 'em',  'name': '能量最小化 (EM)',   'description': '最陡下降最小化以消除空间冲突。',           'status': 'pending'},
                {'id': 'nvt', 'name': 'NVT 平衡',           'description': '恒定粒子数、体积、温度。',                 'status': 'pending'},
                {'id': 'npt', 'name': 'NPT 平衡',           'description': '恒定粒子数、压力、温度。',                 'status': 'pending'},
                {'id': 'md',  'name': '生产模拟',           'description': '生产运行以收集数据。',                     'status': 'pending'},
            ])
        self.tasks[task_id] = {
            'task_id': task_id,
            'params': params,
            'task_type': task_type,
            'title': params.get('description', '') or params.get('pdb_filename', '') or params.get('molecule_filename', ''),
            'submitted_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'pending',
            'progress': 0,
            'paused': False,
            'cancelled': False,
            'current_step': '',
            'logs': [],
            'steps': steps,
            'completed_at': None,
            'results_ready': False,
            'work_dir': os.path.join(self.tasks_dir, task_id),
        }
        self._save_tasks()
        return self.tasks[task_id]

    def _find_cmd(self, name, fallback_paths=None):
        """查找可执行文件路径，支持硬编码兜底路径"""
        cmd = shutil.which(name)
        if not cmd and fallback_paths:
            for fb in fallback_paths:
                if os.path.exists(fb):
                    cmd = fb
                    break
        return cmd

    def _pdb_to_xyz(self, pdb_path, xyz_path):
        """将 PDB 文件转换为 XYZ 格式（无需 obabel）"""
        atoms = []
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith(('ATOM  ', 'HETATM')):
                    try:
                        elem = line[76:78].strip()
                        if not elem:
                            elem = line[12:14].strip()
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        atoms.append((elem, x, y, z))
                    except (ValueError, IndexError):
                        continue
        if not atoms:
            return False
        with open(xyz_path, 'w') as f:
            f.write(f'{len(atoms)}\n')
            f.write('Converted from PDB\n')
            for elem, x, y, z in atoms:
                f.write(f'{elem:2s}  {x:.6f}  {y:.6f}  {z:.6f}\n')
        return True

    def _add_log(self, task_id, message, level='info'):
        """向任务日志中添加一条记录"""
        now = datetime.now()
        time_str = now.strftime('%H:%M:%S')
        # 自动关联当前正在运行的步骤
        step_id = self.tasks[task_id].get('current_step', '')
        self.tasks[task_id]['logs'].append({
            'time': time_str,
            'level': level,
            'message': message,
            'step_id': step_id
        })

    def _update_step(self, task_id, step_id, status, detail=''):
        """更新指定步骤的状态"""
        for step in self.tasks[task_id]['steps']:
            if step['id'] == step_id:
                step['status'] = status
                if detail:
                    step['detail'] = detail

                if status == 'running':
                    self.tasks[task_id]['current_step'] = step_id
                    step['started_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                if status == 'completed':
                    step['completed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    if 'detail' not in step or not step.get('detail'):
                        now = datetime.now()
                        step['detail'] = now.strftime('%H:%M 完成')

        self._save_tasks()

    def _run_shell(self, cmd, task_id, timeout=None, cwd=None):
        """
        执行 shell 命令并实时捕获输出。
        支持暂停（检查 paused 标志）和取消（检查 cancelled 标志）。
        cwd: 指定命令的工作目录（与 topol.top 中 #include 的相对路径有关）
        返回 True 表示成功，False 表示失败或被取消。
        """
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            preexec_fn=os.setsid,  # 允许杀死进程组
            cwd=cwd
        )

        while True:
            # 检查是否被取消
            if self.tasks[task_id]['cancelled']:
                os.killpg(os.getpgid(proc.pid), 15)
                proc.terminate()
                self._add_log(task_id, '模拟已被取消。', 'error')
                return False

            # 检查是否暂停
            if self.tasks[task_id]['paused']:
                os.killpg(os.getpgid(proc.pid), 19)  # SIGSTOP 暂停进程
                while self.tasks[task_id]['paused']:
                    time.sleep(0.5)
                    if self.tasks[task_id]['cancelled']:
                        os.killpg(os.getpgid(proc.pid), 9)
                        proc.terminate()
                        self._add_log(task_id, '模拟已被取消。', 'error')
                        return False
                os.killpg(os.getpgid(proc.pid), 18)  # SIGCONT 恢复进程
                self._add_log(task_id, '模拟已恢复。', 'info')

            # 读取一行输出
            line = proc.stdout.readline()
            if not line:
                break

            line = line.strip()
            if line:
                self._add_log(task_id, line)
                self._parse_progress(task_id, line)

            # 解析完成后 returncode
            ret = proc.poll()
            if ret is not None:
                break

            time.sleep(0.05)

        # 读取剩余输出
        for line in proc.stdout:
            line = line.strip()
            if line:
                self._add_log(task_id, line)
                self._parse_progress(task_id, line)

        proc.wait()
        return proc.returncode == 0

    def _parse_progress(self, task_id, line):
        """从 GROMACS 输出行解析进度百分比"""
        # 匹配 EM 进度: "Step 10: Energy = ..."
        m = re.search(r'Step\s+(\d+):', line)
        if m:
            step_num = int(m.group(1))
            current_step = self.tasks[task_id]['current_step']
            if current_step == 'em':
                progress = min(25, int(step_num / 50000 * 25))
                self.tasks[task_id]['progress'] = progress
                # 更新步骤详情
                for step in self.tasks[task_id]['steps']:
                    if step['id'] == 'em' and step['status'] == 'running':
                        step['progress'] = min(100, int(step_num / 50000 * 100))
                        step['detail'] = f'步骤 {step_num} / 50000'

        # 匹配 MD run 进度: "Step 0   Time 0.000" 或 "Step 500   Time 1.000"
        m = re.search(r'Step\s+(\d+)\s+Time\s+([\d.]+)', line)
        if m:
            step_num = int(m.group(1))
            current_step = self.tasks[task_id]['current_step']
            if current_step in ('nvt', 'npt', 'md'):
                # 估算各阶段的总步骤
                total_steps = {'nvt': 50000, 'npt': 50000, 'md': 500000}  # 默认值
                base_progress = {'nvt': 25, 'npt': 50, 'md': 75}
                total = total_steps.get(current_step, 50000)
                base = base_progress.get(current_step, 0)
                sub_progress = min(100, int(step_num / total * 100))
                self.tasks[task_id]['progress'] = min(100, base + int(sub_progress / 100 * (base_progress.get('md', 0) if current_step == 'md' else 25) or 25))
                for step in self.tasks[task_id]['steps']:
                    if step['id'] == current_step:
                        step['progress'] = sub_progress
                        step['detail'] = f'步骤 {step_num} / {total}'

    def _write_mdp_file(self, task_id, filename, template, replacements=None):
        """写入 .mdp 参数文件"""
        content = template
        if replacements:
            for key, value in replacements.items():
                content = content.replace(key, str(value))
        filepath = os.path.join(self.tasks[task_id]['work_dir'], filename)
        with open(filepath, 'w') as f:
            f.write(content)
        return filepath

    def _run_docking(self, task_id):
        """使用 AutoDock Vina 执行分子对接，返回 True 表示对接成功，False 表示对接失败或跳过"""
        work_dir = self.tasks[task_id]['work_dir']
        params = self.tasks[task_id]['params']

        processed_gro = os.path.join(work_dir, 'processed.gro')
        processed_pdb = os.path.join(work_dir, 'processed.pdb')

        # 检查 vina 和 obabel 是否可用
        vina_cmd = self._find_cmd('vina', ['/usr/local/bin/vina', '/usr/bin/vina'])
        if not vina_cmd:
            vina_cmd = self._find_cmd('vina.exe')
        obabel_cmd = self._find_cmd('obabel', ['/home/marktang/miniconda3/envs/md_env/bin/obabel'])

        if not vina_cmd or not obabel_cmd:
            missing = []
            if not vina_cmd: missing.append('AutoDock Vina')
            if not obabel_cmd: missing.append('Open Babel')
            self._add_log(task_id, f'缺少工具: {", ".join(missing)}，跳过分子对接。', 'error')
            return False

        self._add_log(task_id, f'Vina 路径: {vina_cmd}', 'info')
        self._add_log(task_id, f'Open Babel 路径: {obabel_cmd}', 'info')

        # Step 1: 将 processed.gro 转换为 PDB
        self._add_log(task_id, '正在将处理后的结构转换为 PDB 格式...', 'info')
        cmd = f'echo "System" | {self.gmx} editconf -f {processed_gro} -o {processed_pdb} 2>&1'
        if not self._run_shell(cmd, task_id, cwd=work_dir):
            self._add_log(task_id, 'GRO 转 PDB 失败，跳过对接。', 'error')
            return False

        # Step 2: 将受体 PDB 转换为 PDBQT 格式（-xr = 刚性受体，无可旋转键）
        receptor_pdbqt = os.path.join(work_dir, 'receptor.pdbqt')
        self._add_log(task_id, '正在将受体转换为 PDBQT 格式...', 'info')
        cmd = f'{obabel_cmd} {processed_pdb} -O {receptor_pdbqt} -xr 2>&1'
        if not self._run_shell(cmd, task_id, cwd=work_dir):
            self._add_log(task_id, '受体 PDBQT 转换失败，跳过对接。', 'error')
            return False

        # Step 3: 将配体文件转换为 PDBQT 格式
        ligand_path = params.get('ligand_path')
        _, lig_ext = os.path.splitext(ligand_path)
        ligand_in_work = os.path.join(work_dir, f'ligand{lig_ext}')
        ligand_pdbqt = os.path.join(work_dir, 'ligand.pdbqt')
        self._add_log(task_id, '正在将配体转换为 PDBQT 格式...', 'info')
        cmd = f'{obabel_cmd} {ligand_in_work} -O {ligand_pdbqt} --gen3d 2>&1'
        if not self._run_shell(cmd, task_id, cwd=work_dir):
            self._add_log(task_id, '配体 PDBQT 转换失败，跳过对接。', 'error')
            return False

        # Step 4: 运行 Vina 对接
        center_x = params.get('center_x', 0)
        center_y = params.get('center_y', 0)
        center_z = params.get('center_z', 0)
        size_x = params.get('size_x', 20)
        size_y = params.get('size_y', 20)
        size_z = params.get('size_z', 20)
        exhaustiveness = params.get('exhaustiveness', 8)
        output_pdbqt = os.path.join(work_dir, 'docked_ligand.pdbqt')

        self._add_log(task_id, f'对接参数: 中心 ({center_x}, {center_y}, {center_z})  大小 ({size_x}x{size_y}x{size_z})  精度 {exhaustiveness}', 'info')
        self._add_log(task_id, '正在运行 AutoDock Vina 对接...', 'info')

        cmd = (f'{vina_cmd} --receptor {receptor_pdbqt} '
               f'--ligand {ligand_pdbqt} '
               f'--center_x {center_x} --center_y {center_y} --center_z {center_z} '
               f'--size_x {size_x} --size_y {size_y} --size_z {size_z} '
               f'--exhaustiveness {exhaustiveness} '
               f'--out {output_pdbqt} 2>&1')
        if not self._run_shell(cmd, task_id, cwd=work_dir):
            self._add_log(task_id, 'Vina 对接失败，请检查参数和输入文件。', 'error')
            return False

        # Step 5: 解析 Vina 输出，获取结合能
        self._add_log(task_id, '正在解析对接结果...', 'info')
        docking_results = self._parse_vina_output(task_id, output_pdbqt)
        if docking_results:
            best_mode = docking_results[0]
            self.tasks[task_id]['docking_results'] = {
                'modes': docking_results,
                'best_affinity': best_mode['affinity'],
                'best_rmsd_lb': best_mode['rmsd_lb'],
                'best_rmsd_ub': best_mode['rmsd_ub'],
            }
            self._add_log(task_id, f'最佳结合能: {best_mode["affinity"]} kcal/mol', 'success')
            self._add_log(task_id, f'最佳模式 RMSD: {best_mode["rmsd_lb"]} (l.b.) / {best_mode["rmsd_ub"]} (u.b.)', 'info')

            # Step 6: 将最佳模式的对接结果转换为 PDB
            docked_pdb = os.path.join(work_dir, 'docked_ligand.pdb')
            self._add_log(task_id, '正在将最佳对接模式转换为 PDB 格式...', 'info')
            cmd = f'{obabel_cmd} {output_pdbqt} -O {docked_pdb} -f 1 -l 1 2>&1'
            self._run_shell(cmd, task_id, cwd=work_dir)

            self._save_tasks()
            return True
        else:
            self._add_log(task_id, '未解析到对接结果。', 'warning')
            return False

    def _parse_vina_output(self, task_id, output_pdbqt):
        """解析 Vina 输出的 PDBQT 文件，提取各模式的结合能"""
        results = []
        try:
            with open(output_pdbqt, 'r') as f:
                content = f.read()

            # Vina 将结果写在每个 MODEL 的 REMARK 行中
            # 格式: REMARK VINA RESULT: -7.5 0.000 0.000
            pattern = r'MODEL\s+(\d+).*?REMARK VINA RESULT:\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)'
            matches = re.findall(pattern, content, re.DOTALL)

            for match in matches:
                mode = int(match[0])
                affinity = float(match[1])
                rmsd_lb = float(match[2])
                rmsd_ub = float(match[3])
                results.append({
                    'mode': mode,
                    'affinity': affinity,
                    'rmsd_lb': rmsd_lb,
                    'rmsd_ub': rmsd_ub,
                })

            # 按结合能排序（从低到高，负值越大结合越强）
            results.sort(key=lambda x: x['affinity'])
        except Exception as e:
            self._add_log(task_id, f'解析 Vina 输出时出错: {str(e)}', 'error')

        return results

    def _run_gaussian(self, task_id):
        """使用 Gaussian 进行量子化学计算，返回 True 表示成功"""
        work_dir = self.tasks[task_id]['work_dir']
        params = self.tasks[task_id]['params']
        task_type = self.tasks[task_id].get('task_type', 'md')

        # 查找输入分子文件（兼容 MD 管道和 Gaussian 专用任务）
        mol_path = params.get('molecule_path') or params.get('ligand_path')
        if not mol_path or not os.path.exists(mol_path):
            self._add_log(task_id, '分子文件不存在，跳过 Gaussian 计算。', 'error')
            return False

        # 检查 Gaussian 是否可用（优先 g16，再 g09）
        gaussian_cmd = shutil.which('g16') or shutil.which('g09')
        # 兜底：直接检查常见安装路径
        if not gaussian_cmd:
            for fallback in ['/usr/local/g09/g09', '/usr/local/g16/g16']:
                if os.path.exists(fallback):
                    gaussian_cmd = fallback
                    break
        if not gaussian_cmd:
            self._add_log(task_id, 'Gaussian 未安装或未在 PATH 中。', 'error')
            self._add_log(task_id, '请确认 Gaussian 已安装并配置环境变量（g16 或 g09 命令可用）。', 'info')
            return False

        self._add_log(task_id, f'Gaussian 路径: {gaussian_cmd}', 'info')

        calc_type = params.get('gaussian_calc_type', 'Opt') or params.get('calc_type', 'Opt')
        method = params.get('gaussian_method', 'B3LYP') or params.get('method', 'B3LYP')
        basis = params.get('gaussian_basis', '6-311++G(d,p)') or params.get('basis', '6-311++G(d,p)')
        charge = params.get('gaussian_charge', 0) or params.get('charge', 0)
        multiplicity = params.get('gaussian_multiplicity', 1) or params.get('multiplicity', 1)
        nproc = params.get('gaussian_nproc', 4) or params.get('nproc', 4)
        mem = params.get('gaussian_mem', '4GB') or params.get('mem', '4GB')
        solvent = params.get('solvent', 'water')

        # 自动检测并限制资源使用
        try:
            import subprocess
            result = subprocess.run(['nproc'], capture_output=True, text=True)
            avail_cpus = int(result.stdout.strip())
        except:
            avail_cpus = 2
        if nproc > avail_cpus:
            self._add_log(task_id, f'请求 {nproc} 核，但服务器仅有 {avail_cpus} 核，自动调整为 {avail_cpus} 核。', 'warning')
            nproc = avail_cpus

        try:
            import subprocess
            result = subprocess.run(['free', '-m'], capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')
            parts = lines[1].split()
            avail_mem_mb = int(parts[1])  # total memory in MB
        except:
            avail_mem_mb = 1024
        import re
        mem_mb_match = re.search(r'(\d+)\s*GB', mem, re.IGNORECASE)
        if mem_mb_match:
            req_mem_mb = int(mem_mb_match.group(1)) * 1024
        else:
            mem_mb_match = re.search(r'(\d+)\s*MB', mem, re.IGNORECASE)
            req_mem_mb = int(mem_mb_match.group(1)) if mem_mb_match else 2048
        safe_mem_mb = int(avail_mem_mb * 0.6)
        if req_mem_mb > safe_mem_mb:
            adjusted_mem = f'{safe_mem_mb}MB'
            self._add_log(task_id, f'请求 {mem}，但服务器可用内存为 {avail_mem_mb}MB，自动调整为 {adjusted_mem}。', 'warning')
            mem = adjusted_mem

        # 检查磁盘空间
        try:
            import subprocess
            disk_result = subprocess.run(['df', '-m', work_dir], capture_output=True, text=True)
            disk_lines = disk_result.stdout.strip().split('\n')
            if len(disk_lines) >= 2:
                disk_parts = disk_lines[1].split()
                avail_disk_mb = int(disk_parts[3])
                if avail_disk_mb < 500:
                    self._add_log(task_id, f'警告：磁盘仅剩 {avail_disk_mb}MB 可用空间，Gaussian 需要较多临时文件空间。', 'warning')
                    if avail_disk_mb < 100:
                        self._add_log(task_id, '磁盘空间严重不足，Gaussian 可能无法完成计算！建议清理旧任务文件。', 'error')
                else:
                    self._add_log(task_id, f'磁盘可用空间: {avail_disk_mb}MB', 'info')
        except:
            pass

        self._add_log(task_id, f'计算类型: {calc_type}', 'info')
        self._add_log(task_id, f'方法/基组: {method}/{basis}', 'info')
        self._add_log(task_id, f'电荷/多重度: {charge}/{multiplicity}', 'info')
        self._add_log(task_id, f'CPU/内存: {nproc}核/{mem}', 'info')

        # 检查溶剂模型
        solvent = str(solvent).lower().strip()
        use_solvent = solvent not in ('none', '', 'gas')

        # Step 1: 将分子文件转换为 XYZ 格式（Gaussian 直接支持）
        _, mol_ext = os.path.splitext(mol_path)
        # 查找工作目录中已存在的分子文件
        mol_in_work = os.path.join(work_dir, f'input{mol_ext}')
        if not os.path.exists(mol_in_work):
            mol_in_work = os.path.join(work_dir, f'ligand{mol_ext}')
            if not os.path.exists(mol_in_work):
                self._add_log(task_id, f'工作目录中未找到输入分子文件。', 'error')
                return False

        mol_xyz = os.path.join(work_dir, 'ligand.xyz')

        obabel_cmd = self._find_cmd('obabel', ['/home/marktang/miniconda3/envs/md_env/bin/obabel'])
        if obabel_cmd:
            self._add_log(task_id, f'正在将分子转换为 XYZ 格式...', 'info')
            cmd = f'{obabel_cmd} {mol_in_work} -O {mol_xyz} 2>&1'
            if not self._run_shell(cmd, task_id, cwd=work_dir):
                self._add_log(task_id, 'obabel 转换失败，尝试直接读取分子文件。', 'warning')
                obabel_cmd = None
        if not obabel_cmd:
            self._add_log(task_id, 'Open Babel 未安装或转换失败，尝试直接解析分子文件。', 'warning')
            if mol_ext.lower() == '.pdb':
                self._add_log(task_id, '正在从 PDB 文件中提取原子坐标...', 'info')
                if not self._pdb_to_xyz(mol_in_work, mol_xyz):
                    self._add_log(task_id, 'PDB 坐标提取失败，请检查分子文件或安装 Open Babel。', 'error')
                    return False
            elif mol_ext.lower() == '.xyz':
                shutil.copy2(mol_in_work, mol_xyz)
            else:
                self._add_log(task_id, f'不支持的分子格式 {mol_ext}，请安装 Open Babel 或使用 .pdb/.xyz 格式。', 'error')
                return False

        # Step 2: 读取 XYZ 坐标
        try:
            with open(mol_xyz, 'r') as f:
                xyz_lines = f.readlines()
            atom_count = int(xyz_lines[0].strip())
            # 检查分子大小
            if atom_count > 200:
                self._add_log(task_id, f'警告：当前分子包含 {atom_count} 个原子，Gaussian 量子化学计算不适合大分子。', 'warning')
                self._add_log(task_id, '建议使用 200 个原子以内的小分子（如配体、小分子药物）进行 Gaussian 计算。', 'warning')
                self._add_log(task_id, f'如果仍需继续，请耐心等待，计算可能非常缓慢或失败。', 'warning')
            elif atom_count > 100:
                self._add_log(task_id, f'提示：当前分子包含 {atom_count} 个原子，计算可能需要较长时间。', 'info')
            # 跳过前两行（原子数 + 注释行）
            coord_lines = xyz_lines[2:2+atom_count]

            # 计算时间预估
            try:
                if method in ('PM3', 'AM1', 'PM6', 'PM7'):
                    est_mins = int(atom_count * 0.5 * (2.0 / nproc))
                else:
                    basis_scale = {'3-21G': 1, '6-31G(d)': 2, '6-31+G(d,p)': 3, '6-311++G(d,p)': 5, 'cc-pVDZ': 3, 'cc-pVTZ': 8, 'def2SVP': 2, 'def2TZVP': 8}
                    scale = basis_scale.get(basis, 2)
                    est_mins = int(atom_count * scale * (2.0 / nproc))
                if calc_type in ('Opt', 'Opt+Freq'):
                    est_mins *= 8
                elif calc_type == 'Freq':
                    est_mins *= 2
                if est_mins > 120:
                    est_str = f'约 {est_mins//60}-{est_mins//60+1} 小时'
                else:
                    est_str = f'约 {est_mins} 分钟'
                self._add_log(task_id, f'预估计算时间: {est_str}（{atom_count} 原子 × {method} / {nproc} 核）', 'info')
                if est_mins > 600:
                    self._add_log(task_id, '⚠️ 预估超过 10 小时！建议换用小基组（如 6-31G(d)）或增加 CPU 核心数。', 'warning')
            except:
                pass
        except Exception as e:
            self._add_log(task_id, f'读取分子坐标失败: {str(e)}', 'error')
            return False

        # Step 3: 构建 Gaussian 输入文件
        semi_empirical = method in ('PM3', 'AM1', 'PM6', 'PM7')
        if use_solvent:
            route = f'#p {method}/{basis} {calc_type} SCRF=(Solvent={solvent})' if not semi_empirical else f'#p {method} {calc_type} SCRF=(Solvent={solvent})'
            self._add_log(task_id, f'溶剂模型: {solvent} (SCRF)', 'info')
        else:
            route = f'#p {method}/{basis} {calc_type}' if not semi_empirical else f'#p {method} {calc_type}'
            self._add_log(task_id, '溶剂模型: 气相（无溶剂）', 'info')

        method_display = f'{method}/{basis}' if not semi_empirical else method
        gjf_content = f'%chk=ligand.chk\n'
        gjf_content += f'%nprocshared={nproc}\n'
        gjf_content += f'%mem={mem}\n'
        gjf_content += f'{route}\n\n'
        gjf_content += f'Gaussian calculation - {calc_type} at {method_display}\n\n'
        gjf_content += f'{charge} {multiplicity}\n'
        for line in coord_lines:
            gjf_content += line
        gjf_content += '\n\n'

        gjf_path = os.path.join(work_dir, 'ligand.gjf')
        with open(gjf_path, 'w') as f:
            f.write(gjf_content)

        self._add_log(task_id, 'Gaussian 输入文件已生成，正在运行 Gaussian...', 'info')
        self._add_log(task_id, f'输入文件: ligand.gjf', 'info')

        # Step 4: 运行 Gaussian
        # 用 tee 同时写入文件（供后续解析）和 PIPE（供实时进度读取）
        log_path = os.path.join(work_dir, 'ligand.log')
        cmd = f'bash -c "set -o pipefail; {gaussian_cmd} < {gjf_path} 2>&1 | tee {log_path}"'
        self._add_log(task_id, 'Gaussian 正在运行（量子化学计算通常需要较长时间，请耐心等待）...', 'info')
        if not self._run_shell(cmd, task_id, cwd=work_dir):
            # 检查常见错误并给出友好提示
            if os.path.exists(log_path):
                log_text = open(log_path).read()
                if 'galloc:  could not allocate memory' in log_text:
                    self._add_log(task_id, 'Gaussian 内存不足，请减少 %mem 设置或升级服务器内存。', 'error')
                elif 'The combination of multiplicity' in log_text and 'impossible' in log_text:
                    self._add_log(task_id, '电荷/多重度设置错误：总电子数与所选多重度不匹配。请检查电荷和多重度参数。', 'error')
                    # 尝试提取电子数辅助诊断
                    import re
                    import subprocess
                    elec_match = re.search(r'The combination of multiplicity \d+ and\s+(\d+)\s+electrons', log_text)
                    if elec_match:
                        n_elec = int(elec_match.group(1))
                        suggestion = '奇数' if n_elec % 2 == 1 else '偶数'
                        self._add_log(task_id, f'提示：当前分子共有 {n_elec} 个电子（{suggestion}），多重度应设置为 {2 if n_elec % 2 == 1 else 1}。', 'info')
                elif 'Invalid NProcShared' in log_text or 'Invalid number of processors' in log_text:
                    self._add_log(task_id, 'CPU 核心数设置无效，已自动调整但仍失败。', 'error')
                elif 'segmentation fault' in log_text.lower():
                    self._add_log(task_id, 'Gaussian 段错误（Segmentation Fault），可能内存不足。', 'error')
                elif 'OpenMo"l not found' in log_text or 'could not open' in log_text:
                    self._add_log(task_id, 'Gaussian 找不到输入文件或临时文件夹。', 'error')
                elif 'Erroneous write' in log_text:
                    self._add_log(task_id, 'Gaussian 写入错误，磁盘空间可能不足。', 'error')
                elif 'L123' in log_text and 'cannot' in log_text:
                    self._add_log(task_id, 'Gaussian Link 123 错误，通常与内存/磁盘不足有关。', 'error')
                else:
                    self._add_log(task_id, 'Gaussian 运行失败，请检查输入文件和 Gaussian 安装。', 'error')
            else:
                self._add_log(task_id, 'Gaussian 运行失败，未生成日志文件。', 'error')
            return False

        self._add_log(task_id, 'Gaussian 计算完成，正在解析结果...', 'info')

        # Step 5: 解析 Gaussian 输出
        results = self._parse_gaussian_output(task_id, log_path)
        if results:
            # 保存计算参数供前端展示
            results['method'] = method
            results['basis'] = basis
            results['calc_type'] = calc_type
            results['charge'] = charge
            results['multiplicity'] = multiplicity
            results['nproc'] = nproc
            results['mem'] = mem
            self.tasks[task_id]['gaussian_results'] = results
            self._add_log(task_id, f'最终能量: {results.get("final_energy", "N/A")} Hartree', 'success')

            # Step 6: 如果优化成功，提取优化后的结构
            if results.get('opt_xyz'):
                self._add_log(task_id, '正在保存优化后的配体结构...', 'info')
                opt_xyz_path = os.path.join(work_dir, 'gaussian_opt.xyz')
                opt_pdb_path = os.path.join(work_dir, 'gaussian_opt.pdb')

                with open(opt_xyz_path, 'w') as f:
                    f.write(results['opt_xyz'])

                # 用 obabel 将 XYZ 转 PDB（如果有 obabel）
                if obabel_cmd:
                    cmd = f'{obabel_cmd} {opt_xyz_path} -O {opt_pdb_path} 2>&1'
                    self._run_shell(cmd, task_id, cwd=work_dir)

            # Step 7: 生成 HOMO/LUMO cube 文件（用于 3D 轨道可视化）
            if results.get('normal_termination') and results.get('homo_energy') is not None:
                self._generate_cube_files(task_id, work_dir, gaussian_cmd)

            # 频率结果
            if results.get('frequencies'):
                freq_str = ', '.join([f'{f:.1f}' for f in results['frequencies'][:5]])
                self._add_log(task_id, f'前 5 个频率 (cm⁻¹): {freq_str}', 'info')

            # 轨道能量信息
            if results.get('homo_energy') is not None:
                gap_ev = results.get('gap_energy', 0) * 27.2114
                self._add_log(task_id, f'HOMO: {results["homo_energy"]:.4f} Hartree, LUMO: {results["lumo_energy"]:.4f} Hartree, 能隙: {gap_ev:.2f} eV', 'info')

            self._save_tasks()
            return True
        else:
            self._add_log(task_id, '解析 Gaussian 输出失败，请检查 lignd.log 文件。', 'error')
            return False

    def _parse_gaussian_output(self, task_id, log_path):
        """解析 Gaussian .log 输出文件"""
        results = {}
        try:
            with open(log_path, 'r') as f:
                content = f.read()

            # 检查计算是否正常结束
            results['normal_termination'] = 'Normal termination' in content
            if not results['normal_termination']:
                self._add_log(task_id, 'Gaussian 计算未正常终止。', 'warning')

            # 提取最终能量 - 支持多种方法
            # B3LYP/HF: "SCF Done:  E(RB3LYP) =  -123.456789 A.U."
            # MP2: "EUMP2 =  -123.456789"
            energy = None
            for line in reversed(content.split('\n')):
                if 'SCF Done' in line:
                    parts = line.strip().split()
                    for i, p in enumerate(parts):
                        if p == '=' and i + 1 < len(parts):
                            try:
                                energy = float(parts[i+1])
                                break
                            except:
                                pass
                    if energy is not None:
                        break
                if 'EUMP2' in line and energy is None:
                    parts = line.strip().split()
                    for i, p in enumerate(parts):
                        if p == 'EUMP2' and i + 1 < len(parts):
                            try:
                                energy = float(parts[i+1])
                                break
                            except:
                                pass
                    if energy is not None:
                        break

            if energy is not None:
                results['final_energy'] = energy
                results['final_energy_unit'] = 'Hartree'

            # 提取优化后的结构（"Standard orientation" 最后一个）
            if 'Optimization completed' in content or 'Stationary point found' in content:
                # 找到最后一个 "Standard orientation"
                last_std = content.rfind('Standard orientation')
                if last_std == -1:
                    last_std = content.rfind('Input orientation')

                if last_std != -1:
                    # 从 orientation 往后找到坐标部分
                    section = content[last_std:]
                    lines = section.split('\n')
                    coord_start = -1
                    for i, line in enumerate(lines):
                        if 'Coordinates' in line or '---' in line:
                            coord_start = i + 1
                            break
                    if coord_start == -1 or coord_start >= len(lines):
                        coord_start = 5  # fallback

                    atoms = []
                    for i in range(coord_start, len(lines)):
                        line = lines[i].strip()
                        if not line or '-------' in line:
                            break
                        parts = line.split()
                        if len(parts) >= 6:
                            try:
                                atom_num = int(parts[1])
                                x = float(parts[3])
                                y = float(parts[4])
                                z = float(parts[5])
                                # 原子序数到元素符号
                                elem_map = {1:'H', 6:'C', 7:'N', 8:'O', 9:'F', 15:'P', 16:'S', 17:'Cl',
                                            35:'Br', 53:'I', 5:'B', 14:'Si', 3:'Li', 11:'Na', 19:'K',
                                            12:'Mg', 20:'Ca', 26:'Fe', 30:'Zn'}
                                elem = elem_map.get(atom_num, f'X{atom_num}')
                                atoms.append((elem, x, y, z))
                            except:
                                pass

                    if atoms:
                        # 生成 XYZ 格式
                        xyz = f'{len(atoms)}\nOptimized structure from Gaussian\n'
                        for elem, x, y, z in atoms:
                            xyz += f'{elem} {x:.6f} {y:.6f} {z:.6f}\n'
                        results['opt_xyz'] = xyz
                        results['atom_count'] = len(atoms)

            # 提取频率
            frequencies = []
            for line in content.split('\n'):
                if line.strip().startswith('Frequencies --'):
                    parts = line.strip().split()
                    for p in parts[2:]:
                        try:
                            frequencies.append(float(p))
                        except:
                            pass
            if frequencies:
                results['frequencies'] = frequencies
                # 检查是否有虚频（负频率）
                imag_count = sum(1 for f in frequencies if f < 0)
                if imag_count > 0:
                    results['imaginary_frequencies'] = imag_count
                    self._add_log(task_id, f'警告: 发现 {imag_count} 个虚频（负频率）', 'warning')

            # 提取偶极矩
            for line in content.split('\n'):
                if 'Dipole moment' in line and '=' in line:
                    try:
                        parts = line.strip().split('=')
                        if len(parts) >= 2:
                            results['dipole_moment'] = float(parts[-1].strip().split()[0])
                    except:
                        pass

            # 提取 HOMO/LUMO 轨道能量
            lines = content.split('\n')
            occ_eigenvalues = []
            virt_eigenvalues = []
            for line in lines:
                if 'Alpha  occ. eigenvalues' in line:
                    parts = line.strip().split('--')
                    if len(parts) >= 2:
                        vals = parts[-1].strip().split()
                        for v in vals:
                            try:
                                occ_eigenvalues.append(float(v))
                            except:
                                pass
                elif 'Alpha virt. eigenvalues' in line:
                    parts = line.strip().split('--')
                    if len(parts) >= 2:
                        vals = parts[-1].strip().split()
                        for v in vals:
                            try:
                                virt_eigenvalues.append(float(v))
                            except:
                                pass
                elif 'Beta  occ. eigenvalues' in line:
                    parts = line.strip().split('--')
                    if len(parts) >= 2:
                        vals = parts[-1].strip().split()
                        for v in vals:
                            try:
                                occ_eigenvalues.append(float(v))
                            except:
                                pass
                elif 'Beta virt. eigenvalues' in line:
                    parts = line.strip().split('--')
                    if len(parts) >= 2:
                        vals = parts[-1].strip().split()
                        for v in vals:
                            try:
                                virt_eigenvalues.append(float(v))
                            except:
                                pass

            if occ_eigenvalues:
                results['homo_energy'] = max(occ_eigenvalues)
                results['homo_count'] = len(occ_eigenvalues)
            if virt_eigenvalues:
                results['lumo_energy'] = min(virt_eigenvalues)
                results['lumo_count'] = len(virt_eigenvalues)
            if occ_eigenvalues and virt_eigenvalues:
                results['gap_energy'] = results['lumo_energy'] - results['homo_energy']
                # 收集所有轨道能量用于能级图
                results['orbital_energies'] = {
                    'occupied': occ_eigenvalues,
                    'virtual': virt_eigenvalues
                }

            # 提取振动模式向量（用于前端动画）
            normal_modes = self._parse_normal_modes(content, results.get('atom_count', 0))
            if normal_modes:
                results['normal_modes'] = normal_modes

        except Exception as e:
            self._add_log(task_id, f'解析 Gaussian 输出时出错: {str(e)}', 'error')
            return None

        return results if results else None

    def _parse_normal_modes(self, log_content, atom_count):
        """从 Gaussian .log 提取振动模式向量，用于前端频率动画"""
        if atom_count <= 0:
            return None

        modes = []
        lines = log_content.split('\n')

        # 查找 "Frequencies --" 和对应的位移向量
        # 格式:
        #   Frequencies --  123.45  234.56  345.67
        #   ...
        #   Atom  AN      X      Y      Z        X      Y      Z
        #     1    1    0.00   0.00   0.00    ...
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.strip().startswith('Frequencies --'):
                # 解析这一组的频率值
                parts = line.strip().split()
                freq_vals = []
                for p in parts[2:]:
                    try:
                        freq_vals.append(float(p))
                    except:
                        pass
                if not freq_vals:
                    i += 1
                    continue

                # 跳过 Red. masses, Frc consts, IR Inten 行
                j = i + 1
                while j < len(lines) and ('Red. masses' in lines[j] or 'Frc consts' in lines[j] or 'IR Inten' in lines[j] or 'Raman' in lines[j] or 'Dip. str.' in lines[j] or 'Atom  AN' not in lines[j]):
                    if 'Atom  AN' in lines[j]:
                        break
                    j += 1

                if j >= len(lines) or 'Atom  AN' not in lines[j]:
                    i += 1
                    continue

                # 解析位移向量
                disp_start = j + 1
                displacements = []
                for k in range(disp_start, min(disp_start + atom_count + 5, len(lines))):
                    raw = lines[k].strip()
                    if not raw or '-------' in raw:
                        break
                    cols = raw.split()
                    # 格式: 序号 原子序数 X1 Y1 Z1 X2 Y2 Z2 ...
                    if len(cols) >= 5:
                        try:
                            # 每个原子有 3*N 个位移分量
                            atom_disp = []
                            for m in range(3, min(3 + 3 * len(freq_vals), len(cols))):
                                atom_disp.append(float(cols[m]))
                            if atom_disp:
                                displacements.append(atom_disp)
                        except:
                            continue
                    else:
                        break

                if displacements:
                    # 每个模式是一组向量 [dx, dy, dz] 列表
                    n_modes = len(freq_vals)
                    for m_idx in range(n_modes):
                        mode_vectors = []
                        for atom_disp in displacements:
                            start = m_idx * 3
                            if start + 2 < len(atom_disp):
                                mode_vectors.append([
                                    atom_disp[start],
                                    atom_disp[start + 1],
                                    atom_disp[start + 2]
                                ])
                        if mode_vectors:
                            modes.append({
                                'frequency': freq_vals[m_idx],
                                'vectors': mode_vectors
                            })

                i = j + len(displacements) + 1
            else:
                i += 1

        return modes if modes else None

    def _generate_cube_files(self, task_id, work_dir, gaussian_cmd):
        """生成 HOMO/LUMO cube 文件用于 3D 可视化"""
        g09_root = os.path.dirname(os.path.dirname(gaussian_cmd))
        formchk_cmd = os.path.join(g09_root, 'formchk')
        cubegen_cmd = os.path.join(g09_root, 'cubegen')

        chk_path = os.path.join(work_dir, 'ligand.chk')
        fchk_path = os.path.join(work_dir, 'ligand.fchk')

        if not os.path.exists(chk_path):
            self._add_log(task_id, '未找到 Gaussian 检查点文件 (.chk)，跳过 cube 文件生成。', 'warning')
            return False

        if not os.path.exists(formchk_cmd):
            self._add_log(task_id, '未找到 formchk 工具，跳过 cube 文件生成。', 'warning')
            return False

        if not os.path.exists(cubegen_cmd):
            self._add_log(task_id, '未找到 cubegen 工具，跳过 cube 文件生成。', 'warning')
            return False

        self._add_log(task_id, '正在生成轨道 cube 文件（用于 HOMO/LUMO 3D 可视化）...', 'info')

        # Step 1: formchk - 将 .chk 转为 .fchk
        cmd1 = f'{formchk_cmd} {chk_path} {fchk_path} 2>&1'
        ret1 = self._run_shell(cmd1, task_id, cwd=work_dir)
        if not ret1 or not os.path.exists(fchk_path):
            self._add_log(task_id, 'formchk 转换失败，跳过 cube 生成。', 'warning')
            return False

        # Step 2: cubegen HOMO
        homo_cube = os.path.join(work_dir, 'homo.cube')
        cmd2 = f'{cubegen_cmd} 0 MO=HOMO {fchk_path} {homo_cube} 0 2>&1'
        ret2 = self._run_shell(cmd2, task_id, cwd=work_dir)
        if ret2 and os.path.exists(homo_cube):
            self._add_log(task_id, f'HOMO cube 文件已生成 ({os.path.getsize(homo_cube)/1024:.0f} KB)', 'success')
        else:
            self._add_log(task_id, 'HOMO cube 生成失败（部分分子可能不支持），跳过。', 'warning')

        # Step 3: cubegen LUMO
        lumo_cube = os.path.join(work_dir, 'lumo.cube')
        cmd3 = f'{cubegen_cmd} 0 MO=LUMO {fchk_path} {lumo_cube} 0 2>&1'
        ret3 = self._run_shell(cmd3, task_id, cwd=work_dir)
        if ret3 and os.path.exists(lumo_cube):
            self._add_log(task_id, f'LUMO cube 文件已生成 ({os.path.getsize(lumo_cube)/1024:.0f} KB)', 'success')
        else:
            self._add_log(task_id, 'LUMO cube 生成失败（部分分子可能不支持），跳过。', 'warning')

        return True

    def run_gaussian_only(self, task_id):
        """
        运行 Gaussian 专用计算（不包含 MD 流水线）。
        在独立线程中调用。
        """
        task = self.tasks[task_id]
        work_dir = task['work_dir']
        params = task['params']
        os.makedirs(work_dir, exist_ok=True)

        # 复制分子文件到工作目录
        mol_path = params.get('molecule_path')
        if mol_path and os.path.exists(mol_path):
            ext = params.get('molecule_ext', 'pdb')
            dst = os.path.join(work_dir, f'input.{ext}')
            import shutil
            shutil.copy2(mol_path, dst)
            self._add_log(task_id, f'分子文件已复制到工作目录', 'info')

        # 调用已有的 _run_gaussian 方法
        success = self._run_gaussian(task_id)

        if success:
            task['status'] = 'completed'
            task['progress'] = 100
            task['completed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self._update_step(task_id, 'gaussian', 'completed')
            self._add_log(task_id, 'Gaussian 量子化学计算完成！', 'success')
        else:
            task['status'] = 'failed'
            self._update_step(task_id, 'gaussian', 'failed')
            self._add_log(task_id, 'Gaussian 量子化学计算失败。', 'error')

        self._save_tasks()

    def _adjust_protonation_by_ph(self, pdb_path, ph, task_id):
        """
        根据 pH 值调整 PDB 文件中残基的质子化状态。
        通过重命名残基来指示 GROMACS pdb2gmx 使用正确的质子化模板。
        """
        pH_ranges = {'very_low': (0, 3.5, {'HIS': 'HIP', 'ASP': 'ASH', 'GLU': 'GLH'}),
                    'low': (3.5, 5.5, {'HIS': 'HIP'}),
                    'neutral': (5.5, 8.0, {}),
                    'high': (8.0, 10.0, {'HIS': 'HID'}),
                    'very_high': (10.0, 14.0, {'HIS': 'HID', 'CYS': 'CYM', 'LYS': 'LYN'})}

        mapping = {}
        for range_name, (lo, hi, m) in pH_ranges.items():
            if lo <= ph < hi:
                mapping = m
                break

        if not mapping:
            return

        with open(pdb_path, 'r') as f:
            lines = f.readlines()

        changed = set()
        new_lines = []
        for line in lines:
            if line.startswith(('ATOM', 'HETATM')):
                resname = line[17:20].strip()
                if resname in mapping:
                    new_resname = mapping[resname]
                    line = line[:17] + f'{new_resname:>3}' + line[20:]
                    changed.add(resname)
            new_lines.append(line)

        with open(pdb_path, 'w') as f:
            f.writelines(new_lines)

        if changed:
            details = ', '.join([f'{r}→{mapping[r]}' for r in changed])
            self._add_log(task_id, f'pH {ph}: 残基质子化调整 {details}', 'info')

    def run_pipeline(self, task_id):
        """
        运行完整的 GROMACS 流水线。
        在独立线程中调用。
        """
        task = self.tasks[task_id]
        work_dir = task['work_dir']
        params = task['params']
        pdb_path = params.get('pdb_path')
        force_field = params.get('force_field', 'amber99sb-ildn')
        water_model = params.get('water_model', 'tip3p')
        ref_temp = params.get('temperature', 300)
        sim_time = params.get('simulation_time', 1000)
        time_step = params.get('time_step', 2)
        pressure = params.get('pressure', 1.0)
        ion_type = params.get('ion_type', '')
        ion_conc = params.get('ion_concentration', 0)
        solvate = params.get('solvate', True)

        nsteps = int(sim_time * 1000 / time_step)  # 生产模拟总步数

        os.makedirs(work_dir, exist_ok=True)

        # ==========================================
        # Step 0: 准备输入文件
        # ==========================================
        pdb_in_work = os.path.join(work_dir, 'input.pdb')
        shutil.copy2(pdb_path, pdb_in_work)

        # 复制配体文件（可选）
        ligand_path = params.get('ligand_path')
        ligand_in_work = None
        if ligand_path and os.path.exists(ligand_path):
            _, lig_ext = os.path.splitext(ligand_path)
            ligand_in_work = os.path.join(work_dir, f'ligand{lig_ext}')
            shutil.copy2(ligand_path, ligand_in_work)
            self._add_log(task_id, f'配体文件已复制: {params.get("ligand_filename", "unknown")}', 'info')

        task['status'] = 'running'

        # ==========================================
        # Step 1: pdb2gmx — 生成拓扑
        # ==========================================
        # pdb2gmx 需要一个水模型参数来生成拓扑（即使真空模式也要）
        # 真空模式下前端清空了水模型，这里用 spce 兜底
        pdb2gmx_water = water_model if water_model else 'spce'

        self.tasks[task_id]['started_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._add_log(task_id, f'========== 开始模拟流程 ==========', 'info')
        self._add_log(task_id, f'力场: {force_field}, 水模型: {pdb2gmx_water}, 温度: {ref_temp}K, pH: {params.get("ph", 7.0)}', 'info')
        self._update_step(task_id, 'em', 'running', '正在准备拓扑...')

        # pH 调整：根据 pH 值修改残基质子化状态
        ph = float(params.get('ph', 7.0))
        if ph != 7.0:
            self._add_log(task_id, f'pH={ph} 调整质子化状态...', 'info')
            try:
                self._adjust_protonation_by_ph(pdb_in_work, ph, task_id)
            except Exception as e:
                self._add_log(task_id, f'pH 调整失败（将使用默认质子化）: {e}', 'warning')

        cmd = (f'{self.gmx} pdb2gmx -f {pdb_in_work} '
               f'-o {work_dir}/processed.gro '
               f'-p {work_dir}/topol.top '
               f'-ignh -ff {force_field} -water {pdb2gmx_water} 2>&1')
        self._add_log(task_id, '正在执行 pdb2gmx（生成拓扑）...', 'info')
        if not self._run_shell(cmd, task_id, cwd=work_dir):
            task['status'] = 'failed'
            self._save_tasks()
            return
        self._add_log(task_id, 'pdb2gmx 完成。', 'success')

        # ==========================================
        # Step 1.5: 量子化学计算（Gaussian，可选）
        # 使用 Gaussian 对配体进行量子化学计算
        # ==========================================
        if params.get('gaussian_enabled') and ligand_in_work:
            self._add_log(task_id, '========== 开始量子化学计算 (Gaussian) ==========', 'info')
            self._update_step(task_id, 'gaussian', 'running', '正在运行 Gaussian 计算...')
            gaussian_ok = self._run_gaussian(task_id)
            if gaussian_ok:
                self._update_step(task_id, 'gaussian', 'completed')
                self._add_log(task_id, 'Gaussian 计算完成！', 'success')
                # 如果 Gaussian 生成了优化后的结构，替换配体文件用于后续对接
                gaussian_opt_pdb = os.path.join(work_dir, 'gaussian_opt.pdb')
                if os.path.exists(gaussian_opt_pdb):
                    ligand_in_work = gaussian_opt_pdb
                    self._add_log(task_id, '将使用 Gaussian 优化后的配体结构进行后续对接。', 'info')
            else:
                self._update_step(task_id, 'gaussian', 'failed')
                self._add_log(task_id, 'Gaussian 计算失败，将使用原始配体文件继续。', 'warning')
        else:
            if params.get('gaussian_enabled'):
                self._update_step(task_id, 'gaussian', 'failed', '配体文件缺失')
                self._add_log(task_id, '未找到配体文件，跳过 Gaussian 计算。', 'warning')
            else:
                self._update_step(task_id, 'gaussian', 'cancelled', '用户未启用高斯计算')
                self._add_log(task_id, '用户未启用 Gaussian 计算，跳过。', 'info')

        # ==========================================
        # Step 1.6: 分子对接（可选）
        if params.get('docking_enabled') and ligand_in_work:
            self._add_log(task_id, '========== 开始分子对接 (AutoDock Vina) ==========', 'info')
            self._update_step(task_id, 'docking', 'running', '正在运行分子对接...')
            docking_ok = self._run_docking(task_id)
            if docking_ok:
                self._update_step(task_id, 'docking', 'completed')
                self._add_log(task_id, '分子对接完成！', 'success')
            else:
                self._update_step(task_id, 'docking', 'failed')
                self._add_log(task_id, '分子对接失败，将继续执行 MD 模拟。', 'warning')
        else:
            if params.get('docking_enabled'):
                self._update_step(task_id, 'docking', 'failed', '配体文件缺失')
                self._add_log(task_id, '未找到配体文件，跳过分子对接。', 'warning')
            else:
                self._update_step(task_id, 'docking', 'cancelled', '用户未启用分子对接')
                self._add_log(task_id, '用户未启用分子对接，跳过。', 'info')

        # ==========================================
        # Step 2: editconf — 定义模拟盒子
        # 设置盒子类型为十二面体，分子距盒壁 1.0 nm
        # ==========================================
        self._add_log(task_id, '正在执行 editconf（设置盒子大小）...', 'info')
        cmd = (f'echo "System" | {self.gmx} editconf -f {work_dir}/processed.gro '
               f'-o {work_dir}/boxed.gro '
               f'-bt dodecahedron -d 1.0 2>&1')
        if not self._run_shell(cmd, task_id, cwd=work_dir):
            task['status'] = 'failed'
            self._save_tasks()
            return
        self._add_log(task_id, '盒子设置完成。', 'success')

        # 确定后续步骤使用的输入文件
        if solvate:
            step_input = f'{work_dir}/solvated_ions.gro'
        else:
            step_input = f'{work_dir}/boxed.gro'

        # ==========================================
        # Step 2: solvate — 溶剂化（可选）
        # ==========================================
        if solvate:
            self._add_log(task_id, '正在执行 solvate（添加溶剂）...', 'info')
            cmd = (f'{self.gmx} solvate -cp {work_dir}/boxed.gro '
                   f'-cs spc216.gro '
                   f'-o {work_dir}/solvated.gro '
                   f'-p {work_dir}/topol.top 2>&1')
            if not self._run_shell(cmd, task_id, cwd=work_dir):
                task['status'] = 'failed'
                self._save_tasks()
                return
            self._add_log(task_id, '溶剂化完成。', 'success')

            # ==========================================
            # Step 3: genion — 添加离子
            # ==========================================
            self._add_log(task_id, '正在执行 genion（添加离子）...', 'info')
            ions_mdp = self._write_mdp_file(task_id, 'ions.mdp', IONS_MDP)
            cmd = (f'{self.gmx} grompp -f {ions_mdp} '
                   f'-c {work_dir}/solvated.gro '
                   f'-p {work_dir}/topol.top '
                   f'-o {work_dir}/ions.tpr '
                   f'-maxwarn 5 2>&1')
            if not self._run_shell(cmd, task_id, cwd=work_dir):
                task['status'] = 'failed'
                self._save_tasks()
                return

            echo_input = f'SOL\n'
            cmd = f'echo "{echo_input}" | {self.gmx} genion -s {work_dir}/ions.tpr ' \
                  f'-o {work_dir}/solvated_ions.gro ' \
                  f'-p {work_dir}/topol.top ' \
                  f'-pname NA -nname CL -neutral -conc {ion_conc} 2>&1'
            if not self._run_shell(cmd, task_id, cwd=work_dir):
                task['status'] = 'failed'
                self._save_tasks()
                return
            self._add_log(task_id, '离子添加完成。', 'success')
        else:
            self._add_log(task_id, '用户选择不加水模拟，已跳过溶剂化和加离子步骤。', 'info')

        # ==========================================
        # Step 4: EM — 能量最小化
        # ==========================================
        self._add_log(task_id, '========== 开始能量最小化 (EM) ==========', 'info')
        self._update_step(task_id, 'em', 'running', '正在运行能量最小化...')
        em_mdp_content = em_mdp(solvate=solvate)
        em_mdp_path = self._write_mdp_file(task_id, 'em.mdp', em_mdp_content)
        cmd = (f'{self.gmx} grompp -f {em_mdp_path} '
               f'-c {step_input} '
               f'-p {work_dir}/topol.top '
               f'-o {work_dir}/em.tpr '
               f'-maxwarn 5 2>&1')
        if not self._run_shell(cmd, task_id, cwd=work_dir):
            task['status'] = 'failed'
            self._save_tasks()
            return
        cmd = f'{self.gmx} mdrun -v -deffnm {work_dir}/em 2>&1'
        if not self._run_shell(cmd, task_id, cwd=work_dir):
            task['status'] = 'failed'
            self._save_tasks()
            return
        self._update_step(task_id, 'em', 'completed')
        self.tasks[task_id]['progress'] = 25
        self._add_log(task_id, '能量最小化完成！', 'success')

        # ==========================================
        # Step 5: NVT — 平衡
        # ==========================================
        self._add_log(task_id, '========== 开始 NVT 平衡 ==========', 'info')
        self._update_step(task_id, 'nvt', 'running', '正在运行 NVT 平衡...')
        nvt_mdp_content = nvt_mdp(ref_temp, solvate=solvate)
        nvt_mdp_path = self._write_mdp_file(task_id, 'nvt.mdp', nvt_mdp_content)
        cmd = (f'{self.gmx} grompp -f {nvt_mdp_path} '
               f'-c {work_dir}/em.gro '
               f'-r {work_dir}/em.gro '
               f'-p {work_dir}/topol.top '
               f'-o {work_dir}/nvt.tpr '
               f'-maxwarn 5 2>&1')
        if not self._run_shell(cmd, task_id, cwd=work_dir):
            task['status'] = 'failed'
            self._save_tasks()
            return
        cmd = f'{self.gmx} mdrun -v -deffnm {work_dir}/nvt 2>&1'
        if not self._run_shell(cmd, task_id, cwd=work_dir):
            task['status'] = 'failed'
            self._save_tasks()
            return
        self._update_step(task_id, 'nvt', 'completed')
        self.tasks[task_id]['progress'] = 50
        self._add_log(task_id, 'NVT 平衡完成！', 'success')

        # ==========================================
        # Step 6: NPT — 平衡
        # ==========================================
        self._add_log(task_id, '========== 开始 NPT 平衡 ==========', 'info')
        self._update_step(task_id, 'npt', 'running', '正在运行 NPT 平衡...')
        npt_mdp_content = npt_mdp(ref_temp, solvate=solvate)
        npt_mdp_path = self._write_mdp_file(task_id, 'npt.mdp', npt_mdp_content)
        cmd = (f'{self.gmx} grompp -f {npt_mdp_path} '
               f'-c {work_dir}/nvt.gro '
               f'-r {work_dir}/nvt.gro '
               f'-t {work_dir}/nvt.cpt '
               f'-p {work_dir}/topol.top '
               f'-o {work_dir}/npt.tpr '
               f'-maxwarn 5 2>&1')
        if not self._run_shell(cmd, task_id, cwd=work_dir):
            task['status'] = 'failed'
            self._save_tasks()
            return
        cmd = f'{self.gmx} mdrun -v -deffnm {work_dir}/npt 2>&1'
        if not self._run_shell(cmd, task_id, cwd=work_dir):
            task['status'] = 'failed'
            self._save_tasks()
            return
        self._update_step(task_id, 'npt', 'completed')
        self.tasks[task_id]['progress'] = 75
        self._add_log(task_id, 'NPT 平衡完成！', 'success')

        # ==========================================
        # Step 7: Production MD — 生产模拟
        # ==========================================
        self._add_log(task_id, '========== 开始生产模拟 (MD) ==========', 'info')
        self._update_step(task_id, 'md', 'running', '正在运行生产模拟...')
        md_mdp_content = md_mdp(nsteps, ref_temp, solvate=solvate)
        md_mdp_path = self._write_mdp_file(task_id, 'md.mdp', md_mdp_content)
        cmd = (f'{self.gmx} grompp -f {md_mdp_path} '
               f'-c {work_dir}/npt.gro '
               f'-t {work_dir}/npt.cpt '
               f'-p {work_dir}/topol.top '
               f'-o {work_dir}/md.tpr '
               f'-maxwarn 5 2>&1')
        if not self._run_shell(cmd, task_id, cwd=work_dir):
            task['status'] = 'failed'
            self._save_tasks()
            return
        cmd = f'{self.gmx} mdrun -v -deffnm {work_dir}/md 2>&1'
        if not self._run_shell(cmd, task_id, cwd=work_dir):
            task['status'] = 'failed'
            self._save_tasks()
            return
        self._update_step(task_id, 'md', 'completed')
        self.tasks[task_id]['progress'] = 100
        self._add_log(task_id, '生产模拟完成！', 'success')

        # ==========================================
        # Step 8: 分析
        # ==========================================
        self._add_log(task_id, '========== 开始结果分析 ==========', 'info')
        self._run_analysis(task_id)

        # ==========================================
        # 完成 — 清理步骤状态
        # ==========================================
        # 将所有仍处于 pending/running 的步骤标记为 cancelled（安全兜底）
        for step in task['steps']:
            if step['status'] in ('pending', 'running'):
                step['status'] = 'cancelled'
                step['detail'] = '流水线完成，该步骤未执行'
                if step['id'] in ('docking', 'gaussian'):
                    self._add_log(task_id, f'{step["name"]}: 未执行（流水线已完成）', 'warning')

        task['status'] = 'completed'
        task['completed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        task['results_ready'] = True
        self.tasks[task_id]['progress'] = 100
        self._add_log(task_id, '========== 模拟流程成功完成！==========', 'success')
        self._save_tasks()

    def _run_analysis(self, task_id):
        """MD 完成后运行 GROMACS 分析工具"""
        work_dir = self.tasks[task_id]['work_dir']
        analysis_dir = os.path.join(work_dir, 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)

        # RMSD（需要两个输入：拟合参考组、RMSD计算组，都用 Protein 即组 1）
        self._add_log(task_id, '正在计算 RMSD...', 'info')
        cmd = (f'printf "1\\n1\\n" | {self.gmx} rms -s {work_dir}/md.tpr '
               f'-f {work_dir}/md.xtc '
               f'-o {analysis_dir}/rmsd.xvg '
               f'-tu ns 2>&1')
        self._run_shell(cmd, task_id, cwd=work_dir)

        # 能量（每个选择一行，空行结束）
        self._add_log(task_id, '正在提取能量数据...', 'info')
        cmd = (f'printf "11\\n12\\n13\\n15\\n17\\n22\\n23\\n\\n" | {self.gmx} energy '
               f'-f {work_dir}/md.edr '
               f'-o {analysis_dir}/energy.xvg 2>&1')
        self._run_shell(cmd, task_id, cwd=work_dir)

        # 氢键（需要两组输入：参考组和目标组，都用 Protein 即组 1）
        self._add_log(task_id, '正在计算氢键...', 'info')
        cmd = (f'printf "1\\n1\\n" | {self.gmx} hbond -s {work_dir}/md.tpr '
               f'-f {work_dir}/md.xtc '
               f'-num {analysis_dir}/hbond.xvg 2>&1')
        self._run_shell(cmd, task_id, cwd=work_dir)

        # 回旋半径
        self._add_log(task_id, '正在计算回旋半径...', 'info')
        cmd = (f'printf "1\\n" | {self.gmx} gyrate -s {work_dir}/md.tpr '
               f'-f {work_dir}/md.xtc '
               f'-o {analysis_dir}/gyrate.xvg 2>&1')
        self._run_shell(cmd, task_id, cwd=work_dir)

        # 径向分布函数（参考组=全部蛋白原子，选择组=水氧原子）
        self._add_log(task_id, '正在计算径向分布函数...', 'info')
        cmd = (f'{self.gmx} rdf -s {work_dir}/md.tpr '
               f'-f {work_dir}/md.xtc '
               f'-ref "protein" '
               f'-sel "name OW" '
               f'-o {analysis_dir}/rdf.xvg 2>&1')
        self._run_shell(cmd, task_id, cwd=work_dir)

        self._add_log(task_id, '分析完成！', 'success')

        # 将分析目录打包为 tar.gz，方便前端一键下载
        import tarfile
        tar_path = os.path.join(work_dir, 'analysis.tar.gz')
        with tarfile.open(tar_path, 'w:gz') as tar:
            tar.add(analysis_dir, arcname='analysis')
        self._add_log(task_id, '分析数据已打包。', 'info')

        # 蛋白质-配体相互作用分析（基于原始 PDB 结构）
        self._add_log(task_id, '正在分析蛋白-配体相互作用...', 'info')
        pdb_path = os.path.join(work_dir, 'input.pdb')
        result = analyze_interactions(
            pdb_path,
            log_func=lambda msg, level=None: self._add_log(task_id, msg, level or 'info')
        )
        self.tasks[task_id]['interaction_data'] = result
        self._add_log(task_id, '相互作用分析完成。', 'success')

    def get_results(self, task_id):
        """
        收集分析结果并返回前端需要的格式。
        """
        work_dir = self.tasks[task_id]['work_dir']
        task_type = self.tasks[task_id].get('task_type', 'md')

        # 文件信息（通用部分）
        files = {}
        params = self.tasks[task_id].get('params', {})

        # Gaussian 文件映射（两类任务共享）
        gaussian_file_mapping = {
            'gaussian_log': ('ligand.log', 'Gaussian 日志', f'{work_dir}/ligand.log'),
            'gaussian_gjf': ('ligand.gjf', 'Gaussian 输入', f'{work_dir}/ligand.gjf'),
            'gaussian_opt_pdb': ('gaussian_opt.pdb', 'Gaussian 优化结构', f'{work_dir}/gaussian_opt.pdb'),
            'gaussian_chk': ('ligand.chk', 'Gaussian 检查点', f'{work_dir}/ligand.chk'),
            'gaussian_homo_cube': ('homo.cube', 'HOMO 轨道密度', f'{work_dir}/homo.cube'),
            'gaussian_lumo_cube': ('lumo.cube', 'LUMO 轨道密度', f'{work_dir}/lumo.cube'),
            'gaussian_fchk': ('ligand.fchk', '格式化检查点', f'{work_dir}/ligand.fchk'),
        }

        if task_type == 'gaussian':
            # ==========================================
            # Gaussian 专用任务：跳过 MD 分析
            # ==========================================
            for key, (name, _, fpath) in gaussian_file_mapping.items():
                if os.path.exists(fpath):
                    fsize = os.path.getsize(fpath)
                    size_str = f'{fsize / 1024:.0f} KB' if fsize < 1024*1024 else f'{fsize / 1024 / 1024:.1f} MB'
                    files[key] = {
                        'name': name,
                        'size': size_str,
                        'url': f'/download/{task_id}/{key}'
                    }

            # 添加输入分子文件
            mol_ext = params.get('molecule_ext', 'pdb')
            mol_in_work = os.path.join(work_dir, f'input.{mol_ext}')
            if os.path.exists(mol_in_work):
                fsize = os.path.getsize(mol_in_work)
                size_str = f'{fsize / 1024:.0f} KB' if fsize < 1024*1024 else f'{fsize / 1024 / 1024:.1f} MB'
                files['input_mol'] = {
                    'name': params.get('molecule_filename', f'input.{mol_ext}'),
                    'size': size_str,
                    'url': f'/download/{task_id}/input_mol'
                }

            return {
                'task_id': task_id,
                'task_type': 'gaussian',
                'title': self.tasks[task_id].get('title', ''),
                'completed_at': self.tasks[task_id].get('completed_at', ''),
                'metrics': {},
                'charts': {},
                'interactions': {
                    'has_ligand': False,
                    'hbonds': {'count': 0, 'details': [], 'per_residue': []},
                    'hydrophobic': {'count': 0, 'details': [], 'per_residue': []},
                },
                'docking': None,
                'gaussian': self.tasks[task_id].get('gaussian_results', None),
                'files': files
            }

        # ==========================================
        # MD 任务：正常收集分析结果
        # ==========================================
        analysis_dir = os.path.join(work_dir, 'analysis')

        # 解析 RMSD
        rmsd_data = parse_xvg(os.path.join(analysis_dir, 'rmsd.xvg'))

        # 解析能量（3 列：势能、动能、总能量）
        energy_data = parse_xvg(os.path.join(analysis_dir, 'energy.xvg'), columns=6)

        # 解析氢键
        hbond_data = parse_xvg(os.path.join(analysis_dir, 'hbond.xvg'))

        # 解析回旋半径
        rg_data = parse_xvg(os.path.join(analysis_dir, 'gyrate.xvg'))

        # 解析 RDF
        rdf_data = parse_xvg(os.path.join(analysis_dir, 'rdf.xvg'))

        # 计算指标
        rmsd_values = rmsd_data.get('col_0', [])
        energy_pot = energy_data.get('col_0', [])
        rg_values = rg_data.get('col_0', [])
        hbond_values = hbond_data.get('col_0', [])

        metrics = {
            'rmsd': round(sum(rmsd_values) / len(rmsd_values), 2) if rmsd_values else 0,
            'potential_energy': round(energy_pot[-1], 0) if energy_pot else 0,
            'rg': round(sum(rg_values) / len(rg_values), 1) if rg_values else 0,
            'hbonds': round(sum(hbond_values) / len(hbond_values), 0) if hbond_values else 0,
        }

        # MD 文件映射
        md_file_mapping = {
            'xtc': ('trajectory.xtc', '轨迹文件', f'{work_dir}/md.xtc'),
            'log': ('md.log', '日志文件', f'{work_dir}/md.log'),
            'top': ('topol.top', '拓扑文件', f'{work_dir}/topol.top'),
            'gro': ('final.gro', '结构文件', f'{work_dir}/md.gro'),
            'edr': ('energy.edr', '能量文件', f'{work_dir}/md.edr'),
            'xvg': ('analysis.tar.gz', '分析数据', f'{work_dir}/analysis.tar.gz'),
            'pdb': ('input.pdb', '原始 PDB', f'{work_dir}/input.pdb'),
            'docked_pdb': ('docked_ligand.pdb', '对接配体 PDB', f'{work_dir}/docked_ligand.pdb'),
            'docked_pdbqt': ('docked_ligand.pdbqt', '对接配体 PDBQT', f'{work_dir}/docked_ligand.pdbqt'),
            'receptor_pdbqt': ('receptor.pdbqt', '受体 PDBQT', f'{work_dir}/receptor.pdbqt'),
        }

        for key, (name, _, fpath) in md_file_mapping.items():
            if os.path.exists(fpath):
                fsize = os.path.getsize(fpath)
                size_str = f'{fsize / 1024:.0f} KB' if fsize < 1024*1024 else f'{fsize / 1024 / 1024:.1f} MB'
                files[key] = {
                    'name': name,
                    'size': size_str,
                    'url': f'/download/{task_id}/{key}'
                }

        for key, (name, _, fpath) in gaussian_file_mapping.items():
            if os.path.exists(fpath):
                fsize = os.path.getsize(fpath)
                size_str = f'{fsize / 1024:.0f} KB' if fsize < 1024*1024 else f'{fsize / 1024 / 1024:.1f} MB'
                files[key] = {
                    'name': name,
                    'size': size_str,
                    'url': f'/download/{task_id}/{key}'
                }

        # 添加配体文件到文件映射
        lig_path = params.get('ligand_path')
        if lig_path and os.path.exists(lig_path):
            _, lig_ext = os.path.splitext(lig_path)
            lig_in_work = os.path.join(os.path.dirname(work_dir), f'{task_id}_ligand{lig_ext}')
            if os.path.exists(lig_in_work):
                fsize = os.path.getsize(lig_in_work)
                size_str = f'{fsize / 1024:.0f} KB' if fsize < 1024*1024 else f'{fsize / 1024 / 1024:.1f} MB'
                files['ligand'] = {
                    'name': params.get('ligand_filename', 'ligand.xyz'),
                    'size': size_str,
                    'url': f'/download/{task_id}/ligand'
                }

        return {
            'task_id': task_id,
            'task_type': 'md',
            'title': self.tasks[task_id].get('title', ''),
            'completed_at': self.tasks[task_id].get('completed_at', ''),
            'metrics': metrics,
            'charts': {
                'rmsd': {
                    'time': rmsd_data.get('time', []),
                    'rmsd': rmsd_data.get('col_0', [])
                },
                'energy': {
                    'time': energy_data.get('time', []),
                    'potential': energy_data.get('col_0', []),
                    'kinetic': energy_data.get('col_1', []),
                    'total': energy_data.get('col_2', []),
                },
                'hbond': {
                    'time': hbond_data.get('time', []),
                    'hbonds': hbond_data.get('col_0', [])
                },
                'rg': {
                    'time': rg_data.get('time', []),
                    'rg': rg_data.get('col_0', [])
                },
                'rdf': {
                    'r': [x * 10 for x in rdf_data.get('time', [])],
                    'g_r': rdf_data.get('col_0', [])
                }
            },
            'interactions': self.tasks[task_id].get('interaction_data', {
                'has_ligand': False,
                'hbonds': {'count': 0, 'details': [], 'per_residue': []},
                'hydrophobic': {'count': 0, 'details': [], 'per_residue': []},
            }),
            'docking': self.tasks[task_id].get('docking_results', None),
            'gaussian': self.tasks[task_id].get('gaussian_results', None),
            'files': files
        }
