import os
import re
import uuid
import shutil
import threading
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory, abort, Response
from flask_cors import CORS

from gromacs_runner import GromacsRunner
from ai_assistant import chat as ai_chat


def _format_elapsed(seconds):
    """将秒数转换为可读的耗时文本"""
    if seconds is None:
        return None
    seconds = int(seconds)
    if seconds < 60:
        return f'{seconds}秒'
    elif seconds < 3600:
        return f'{seconds // 60}分{seconds % 60}秒'
    else:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f'{h}小时{m}分{s}秒'


def _calc_elapsed(started_at, completed_at=None):
    """计算两个 ISO 时间戳之间的秒数"""
    if not started_at:
        return None
    fmt = '%Y-%m-%d %H:%M:%S'
    try:
        start = datetime.strptime(started_at, fmt)
        end = datetime.strptime(completed_at, fmt) if completed_at else datetime.now()
        return (end - start).total_seconds()
    except (ValueError, TypeError):
        return None

# ====================================================
# Flask 应用初始化
# ====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.dirname(BASE_DIR)
if not os.path.exists(os.path.join(STATIC_DIR, 'index.html')):
    STATIC_DIR = BASE_DIR
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='')
CORS(app)  # 允许来自前端（可能跨域）的请求

# ====================================================
# 配置
# ====================================================
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['TASKS_FOLDER'] = os.path.join(os.path.dirname(__file__), 'tasks')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 最大上传 100 MB

# 确保目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TASKS_FOLDER'], exist_ok=True)

# ====================================================
# GROMACS 管理器（全局单例）
# 自动检测 GROMACS 路径：优先用远程服务器路径，其次 PATH
# ====================================================
import shutil, glob as _glob

_gmx_candidates = [
    '/usr/local/gromacs/bin/gmx',   # 远程服务器路径
    '/usr/local/gromacs/bin/gmx_d', # 调试版
]
_gmx_found = None
for _c in _gmx_candidates:
    if os.path.exists(_c):
        _gmx_found = _c
        break
if not _gmx_found:
    _gmx_found = shutil.which('gmx') or shutil.which('gmx_mpi') or 'gmx'

gromacs = GromacsRunner(app.config['TASKS_FOLDER'], gmx_command=_gmx_found)


# ====================================================
# 接口 8：删除任务
# DELETE /tasks/<task_id>
# ====================================================
@app.route('/tasks/<task_id>', methods=['DELETE'])
def delete_task(task_id):
    """删除任务记录和所有相关文件"""
    success, msg = gromacs.delete_task(task_id)

    # 也删除上传的 PDB 文件
    upload_path = os.path.join(gromacs.tasks_dir, task_id, f'{task_id}.pdb')
    if os.path.exists(upload_path):
        try:
            os.remove(upload_path)
        except Exception:
            pass

    if success:
        return jsonify({'message': msg}), 200
    else:
        return jsonify({'error': msg}), 404

# ====================================================
# 接口 1：上传 PDB 文件
# POST /upload-pdb
# 参数：multipart/form-data，文件 key="pdb"
#       其他字段：force_field, water_model, temperature, simulation_time 等
# 返回：{ "task_id": "xxx", "message": "上传成功" }
# ====================================================
@app.route('/upload-pdb', methods=['POST'])
def upload_pdb():
    """接收 PDB 文件并创建新任务"""

    # 检查是否有文件
    if 'pdb' not in request.files:
        return jsonify({'error': '未找到 PDB 文件，请确保文件字段名称为 "pdb"'}), 400

    file = request.files['pdb']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    # 校验文件扩展名
    if not file.filename.lower().endswith('.pdb'):
        return jsonify({'error': '仅支持 .pdb 格式文件'}), 400

    # 生成唯一任务 ID（格式：分子名_日期_时间）
    pdb_name = os.path.splitext(file.filename)[0]
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', pdb_name)[:20].strip('_') or 'protein'
    task_id = f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 保存上传的 PDB 文件
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{task_id}.pdb')
    file.save(upload_path)

    # 处理配体小分子文件（可选）
    ligand_path = None
    ligand_filename = None
    if 'ligand' in request.files:
        lig_file = request.files['ligand']
        if lig_file and lig_file.filename:
            ligand_filename = lig_file.filename
            ext = ligand_filename.rsplit('.', 1)[-1].lower()
            if ext in ('xyz', 'mol2', 'sdf', 'pdb'):
                ligand_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{task_id}_ligand.{ext}')
                lig_file.save(ligand_path)

    # 收集所有表单参数
    params = {
        'pdb_path': upload_path,
        'pdb_filename': file.filename,
        'ligand_path': ligand_path,
        'ligand_filename': ligand_filename if ligand_path else None,
        'force_field': request.form.get('force_field', 'amber99sb-ildn'),
        'ph': float(request.form.get('ph', 7.0)),
        'water_model': request.form.get('water_model', 'tip3p'),
        'ion_type': request.form.get('ion_type', 'NA'),
        'ion_concentration': float(request.form.get('ion_concentration', 0)),
        'temperature': float(request.form.get('temperature', 300)),
        'simulation_time': int(request.form.get('simulation_time', 1000)),
        'time_step': float(request.form.get('time_step', 2)),
        'pressure': float(request.form.get('pressure', 1.0)),
        'description': request.form.get('description', ''),
        'solvate': request.form.get('solvate', 'true') == 'true',
        'docking_enabled': request.form.get('docking_enabled', 'false') == 'true',
        'center_x': float(request.form.get('center_x', 0)),
        'center_y': float(request.form.get('center_y', 0)),
        'center_z': float(request.form.get('center_z', 0)),
        'size_x': float(request.form.get('size_x', 20)),
        'size_y': float(request.form.get('size_y', 20)),
        'size_z': float(request.form.get('size_z', 20)),
        'exhaustiveness': int(request.form.get('exhaustiveness', 8)),
        'gaussian_enabled': request.form.get('gaussian_enabled', 'false') == 'true',
        'gaussian_calc_type': request.form.get('gaussian_calc_type', 'Opt'),
        'gaussian_method': request.form.get('gaussian_method', 'B3LYP'),
        'gaussian_basis': request.form.get('gaussian_basis', '6-311++G(d,p)'),
        'gaussian_charge': int(request.form.get('gaussian_charge', 0)),
        'gaussian_multiplicity': int(request.form.get('gaussian_multiplicity', 1)),
        'gaussian_nproc': int(request.form.get('gaussian_nproc', 4)),
        'gaussian_mem': request.form.get('gaussian_mem', '4GB'),
    }

    # 创建任务并保存参数
    gromacs.create_task(task_id, params, task_type='md')

    # 保存参数到任务目录
    task_dir = os.path.join(app.config['TASKS_FOLDER'], task_id)
    os.makedirs(task_dir, exist_ok=True)
    with open(os.path.join(task_dir, 'params.json'), 'w') as f:
        import json
        json.dump(params, f, indent=2)

    return jsonify({
        'task_id': task_id,
        'message': '文件上传成功，任务已创建'
    }), 201


# ====================================================
# 接口 1b：上传分子文件（高斯专用）
# POST /upload-gaussian
# ====================================================
@app.route('/upload-gaussian', methods=['POST'])
def upload_gaussian():
    """接收分子文件（或从已有任务引用分子）并创建 Gaussian 专用任务"""

    source_task_id = request.form.get('source_task_id', '')

    if source_task_id:
        # ===== 从已有 MD 任务获取配体分子 =====
        if source_task_id not in gromacs.tasks:
            return jsonify({'error': '源任务不存在'}), 404
        source_task = gromacs.tasks[source_task_id]
        ligand_path = source_task.get('params', {}).get('ligand_path', '')
        if not ligand_path or not os.path.exists(ligand_path):
            return jsonify({'error': '源任务没有可用的配体分子文件，请确认该任务已上传配体'}), 400

        ext = os.path.splitext(ligand_path)[1].lower().lstrip('.')
        mol_basename = os.path.splitext(os.path.basename(ligand_path))[0]
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', mol_basename)[:20].strip('_') or 'ligand'
        task_id = f"gauss_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 复制分子文件到上传文件夹
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{task_id}.{ext}')
        import shutil
        shutil.copy2(ligand_path, upload_path)

        source_desc = source_task.get('params', {}).get('description', '') or source_task_id
        molecule_filename = os.path.basename(ligand_path)
    else:
        # ===== 传统方式：上传新分子文件 =====
        if 'molecule' not in request.files:
            return jsonify({'error': '未找到分子文件，请确保文件字段名称为 "molecule"'}), 400

        file = request.files['molecule']
        if file.filename == '':
            return jsonify({'error': '未选择文件'}), 400

        ext = file.filename.rsplit('.', 1)[-1].lower()
        if ext not in ('pdb', 'xyz', 'mol2', 'sdf'):
            return jsonify({'error': f'不支持的文件格式 .{ext}，支持 .pdb .xyz .mol2 .sdf'}), 400

        # 生成任务 ID
        mol_name = os.path.splitext(file.filename)[0]
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', mol_name)[:20].strip('_') or 'molecule'
        task_id = f"gauss_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 保存分子文件
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{task_id}.{ext}')
        file.save(upload_path)

        source_desc = ''
        molecule_filename = file.filename

    # 收集 Gaussian 参数
    params = {
        'molecule_path': upload_path,
        'molecule_filename': molecule_filename,
        'molecule_ext': ext,
        'calc_type': request.form.get('calc_type', 'Opt'),
        'method': request.form.get('method', 'B3LYP'),
        'basis': request.form.get('basis', '6-311++G(d,p)'),
        'charge': int(request.form.get('charge', 0)),
        'multiplicity': int(request.form.get('multiplicity', 1)),
        'nproc': int(request.form.get('nproc', 4)),
        'mem': request.form.get('mem', '4GB'),
        'solvent': request.form.get('solvent', 'water'),
        'description': request.form.get('description', '').strip() or molecule_filename,
    }
    if source_task_id:
        params['source_task_id'] = source_task_id
        if not request.form.get('description', '').strip():
            params['description'] = f"[从 {source_desc} 引用] {molecule_filename}"

    # 创建 Gaussian 专用任务
    gromacs.create_task(task_id, params, task_type='gaussian')

    # 保存参数
    task_dir = os.path.join(app.config['TASKS_FOLDER'], task_id)
    os.makedirs(task_dir, exist_ok=True)
    with open(os.path.join(task_dir, 'params.json'), 'w') as f:
        import json
        json.dump(params, f, indent=2)

    return jsonify({
        'task_id': task_id,
        'message': '分子文件上传成功，高斯任务已创建'
    }), 201


# ====================================================
# 接口 2：启动 GROMACS 模拟
# POST /run-gromacs
# 参数：application/json，{ "task_id": "xxx" }
# 返回：{ "message": "模拟已启动" }
# ====================================================
@app.route('/run-gromacs', methods=['POST'])
def run_gromacs():
    """在后台线程中启动 GROMACS 流水线"""

    data = request.get_json()
    if not data or 'task_id' not in data:
        return jsonify({'error': '请提供 task_id'}), 400

    task_id = data['task_id']

    if task_id not in gromacs.tasks:
        return jsonify({'error': f'任务 {task_id} 不存在'}), 404

    task = gromacs.tasks[task_id]

    if task['status'] not in ('pending', 'paused'):
        return jsonify({'error': f'任务状态为 {task["status"]}，无法启动'}), 400

    # 在后台线程中运行流水线，避免阻塞 API 响应
    thread = threading.Thread(
        target=gromacs.run_pipeline,
        args=(task_id,),
        daemon=True
    )
    thread.start()

    return jsonify({
        'message': '模拟已启动',
        'task_id': task_id
    })


# ====================================================
# 接口 2b：启动 Gaussian 计算
# POST /run-gaussian
# ====================================================
@app.route('/run-gaussian', methods=['POST'])
def run_gaussian():
    """在后台线程中启动 Gaussian 专用计算"""

    data = request.get_json()
    if not data or 'task_id' not in data:
        return jsonify({'error': '请提供 task_id'}), 400

    task_id = data['task_id']

    if task_id not in gromacs.tasks:
        return jsonify({'error': f'任务 {task_id} 不存在'}), 404

    task = gromacs.tasks[task_id]

    if task['status'] not in ('pending', 'paused'):
        return jsonify({'error': f'任务状态为 {task["status"]}，无法启动'}), 400

    # 立即设置为 running，防止重复提交
    task['status'] = 'running'
    task['started_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    gromacs._update_step(task_id, 'gaussian', 'running')
    gromacs._add_log(task_id, 'Gaussian 量子化学计算开始...', 'info')
    gromacs._save_tasks()

    thread = threading.Thread(
        target=gromacs.run_gaussian_only,
        args=(task_id,),
        daemon=True
    )
    thread.start()

    return jsonify({
        'message': 'Gaussian 计算已启动',
        'task_id': task_id
    })


# ====================================================
# 接口 3：获取任务状态
# GET /tasks/{task_id}/status
# 返回：{ task_id, status, progress, steps: [...] }
# ====================================================
@app.route('/tasks/<task_id>/status', methods=['GET'])
def get_task_status(task_id):
    """返回任务的当前进度、步骤状态等信息"""

    if task_id not in gromacs.tasks:
        return jsonify({'error': '任务不存在'}), 404

    task = gromacs.tasks[task_id]
    params = task['params']

    # 构建步骤信息
    steps = []
    for step in task['steps']:
        elapsed_secs = _calc_elapsed(step.get('started_at'), step.get('completed_at'))
        step_info = {
            'id': step['id'],
            'name': step['name'],
            'description': step['description'],
            'status': step['status'],
            'elapsed': _format_elapsed(elapsed_secs),
            'elapsed_secs': elapsed_secs,
        }
        if step['status'] == 'running' and 'progress' in step:
            step_info['progress'] = step['progress']
        if step.get('detail'):
            step_info['detail'] = step['detail']
        steps.append(step_info)

    # 计算总用时
    total_elapsed_secs = _calc_elapsed(task.get('started_at'), task.get('completed_at'))

    return jsonify({
        'task_id': task_id,
        'title': task.get('title', ''),
        'description': params.get('description', ''),
        'submitted_at': task.get('submitted_at', ''),
        'status': task['status'],
        'progress': task['progress'],
        'steps': steps,
        'elapsed': _format_elapsed(total_elapsed_secs),
        'elapsed_secs': total_elapsed_secs,
    })


# ====================================================
# 接口 4：获取任务日志
# GET /tasks/{task_id}/logs?since=HH:MM:SS
# 返回：{ logs: [{ time, level, message }] }
# since 参数实现增量拉取
# ====================================================
@app.route('/tasks/<task_id>/logs', methods=['GET'])
def get_task_logs(task_id):
    """返回任务的运行日志，支持增量拉取"""

    if task_id not in gromacs.tasks:
        return jsonify({'error': '任务不存在'}), 404

    since = request.args.get('since', '')

    logs = gromacs.tasks[task_id]['logs']

    # 如果传了 since 参数，只返回该时间之后的日志
    if since:
        logs = [log for log in logs if log['time'] >= since]

    return jsonify({
        'logs': logs
    })


# ====================================================
# 接口 4.5：下载任务日志（纯文本文件）
# GET /tasks/{task_id}/logs/download
# 返回 text/plain 文件，供用户保存到本地
# ====================================================
@app.route('/tasks/<task_id>/logs/download', methods=['GET'])
def download_task_logs(task_id):
    """将任务的所有日志导出为可下载的文本文件"""

    if task_id not in gromacs.tasks:
        return jsonify({'error': '任务不存在'}), 404

    logs = gromacs.tasks[task_id]['logs']
    task = gromacs.tasks[task_id]

    # 组装日志内容
    lines = []
    lines.append(f"任务 ID: {task_id}")
    lines.append(f"标题: {task.get('title', '-')}")
    lines.append(f"提交时间: {task.get('submitted_at', '-')}")
    lines.append(f"完成时间: {task.get('completed_at', '-')}")
    lines.append(f"最终状态: {task.get('status', '-')}")
    lines.append(f"总进度: {task.get('progress', 0)}%")
    lines.append("=" * 60)
    lines.append("")

    if not logs:
        lines.append("（暂无日志记录）")
    else:
        for log in logs:
            lines.append(f"[{log['time']}] [{log['level'].upper()}] {log['message']}")

    lines.append("")
    lines.append("=" * 60)
    lines.append(f"共 {len(logs)} 条日志记录")
    lines.append("文件生成时间: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    content = '\n'.join(lines)

    return Response(
        content,
        mimetype='text/plain; charset=utf-8',
        headers={
            'Content-Disposition': f'attachment; filename="{task_id}_logs.txt"',
            'Content-Type': 'text/plain; charset=utf-8'
        }
    )


# ====================================================
# 接口 5：暂停模拟
# POST /tasks/{task_id}/pause
# ====================================================
@app.route('/tasks/<task_id>/pause', methods=['POST'])
def pause_task(task_id):
    """暂停正在运行的模拟任务"""

    if task_id not in gromacs.tasks:
        return jsonify({'error': '任务不存在'}), 404

    task = gromacs.tasks[task_id]

    if task['status'] != 'running':
        return jsonify({'error': f'任务状态为 {task["status"]}，无法暂停'}), 400

    task['paused'] = True
    task['status'] = 'paused'
    gromacs._add_log(task_id, '模拟已暂停。', 'warning')

    return jsonify({'message': '模拟已暂停'})


# ====================================================
# 接口 6：恢复模拟
# POST /tasks/{task_id}/resume
# ====================================================
@app.route('/tasks/<task_id>/resume', methods=['POST'])
def resume_task(task_id):
    """恢复被暂停的模拟任务"""

    if task_id not in gromacs.tasks:
        return jsonify({'error': '任务不存在'}), 404

    task = gromacs.tasks[task_id]

    if task['status'] != 'paused':
        return jsonify({'error': f'任务状态为 {task["status"]}，无法恢复'}), 400

    task['paused'] = False
    task['status'] = 'running'
    gromacs._add_log(task_id, '模拟已恢复。', 'info')

    return jsonify({'message': '模拟已恢复'})


# ====================================================
# 接口 7：取消模拟
# POST /tasks/{task_id}/cancel
# ====================================================
@app.route('/tasks/<task_id>/cancel', methods=['POST'])
def cancel_task(task_id):
    """取消正在运行或暂停的模拟任务"""

    if task_id not in gromacs.tasks:
        return jsonify({'error': '任务不存在'}), 404

    task = gromacs.tasks[task_id]

    if task['status'] not in ('running', 'paused', 'pending'):
        return jsonify({'error': f'任务状态为 {task["status"]}，无法取消'}), 400

    task['cancelled'] = True
    task['status'] = 'cancelled'
    gromacs._add_log(task_id, '模拟已被用户取消。', 'error')

    return jsonify({'message': '模拟已取消'})


# ====================================================
# 接口 8：获取任务列表
# GET /tasks?limit=5
# 返回：{ tasks: [{ task_id, description, status, submitted_at }] }
# ====================================================
@app.route('/tasks', methods=['GET'])
def list_tasks():
    """返回最近的任务列表"""

    limit = request.args.get('limit', 10, type=int)

    tasks_list = []
    for task_id, task in reversed(list(gromacs.tasks.items())):
        params = task.get('params', {})
        # 判断是否有配体分子
        has_ligand = bool(params.get('ligand_path'))
        # 获取分子文件名
        molecule_filename = ''
        if has_ligand:
            molecule_filename = os.path.basename(params.get('ligand_path', ''))
        elif params.get('molecule_filename'):
            molecule_filename = params.get('molecule_filename', '')
        # 判断高斯是否已跑过
        gaussian_ran = False
        steps = task.get('steps', [])
        for s in steps:
            if s.get('id') == 'gaussian' and s.get('status') == 'completed':
                gaussian_ran = True
                break
        tasks_list.append({
            'task_id': task_id,
            'description': params.get('description', '') or task.get('title', ''),
            'status': task['status'],
            'submitted_at': task.get('submitted_at', ''),
            'progress': task['progress'],
            'task_type': task.get('task_type', 'md'),
            'has_ligand': has_ligand,
            'molecule_filename': molecule_filename,
            'gaussian_ran': gaussian_ran,
        })
        if len(tasks_list) >= limit:
            break

    return jsonify({'tasks': tasks_list})


# ====================================================
# 接口 9：获取模拟结果（含指标、图表数据、文件）
# GET /tasks/{task_id}/results
# ====================================================
@app.route('/tasks/<task_id>/results', methods=['GET'])
def get_task_results(task_id):
    """返回已完成任务的模拟结果"""

    if task_id not in gromacs.tasks:
        return jsonify({'error': '任务不存在'}), 404

    task = gromacs.tasks[task_id]

    # 如果任务还没完成，返回空结果
    if task['status'] != 'completed':
        return jsonify({
            'task_id': task_id,
            'title': task.get('title', ''),
            'completed_at': task.get('completed_at'),
            'metrics': {},
            'charts': {},
            'files': {},
            'message': '任务尚未完成'
        })

    # 调用 runner 收集分析结果
    results = gromacs.get_results(task_id)

    # 加入用时信息
    started_at = task.get('started_at', '') or task.get('submitted_at', '')
    results['started_at'] = started_at
    total_elapsed_secs = _calc_elapsed(started_at, task.get('completed_at'))
    results['elapsed'] = _format_elapsed(total_elapsed_secs)
    results['elapsed_secs'] = total_elapsed_secs

    # 加入各步骤用时
    step_timing = []
    for step in task['steps']:
        step_status = step.get('status', '')
        step_secs = _calc_elapsed(step.get('started_at'), step.get('completed_at'))
        # 非 completed 状态不显示耗时
        step_elapsed = _format_elapsed(step_secs) if step_status == 'completed' else None
        step_timing.append({
            'id': step['id'],
            'name': step['name'],
            'started_at': step.get('started_at', ''),
            'completed_at': step.get('completed_at', ''),
            'elapsed': step_elapsed,
            'elapsed_secs': step_secs if step_status == 'completed' else None,
            'status': step_status,
            'detail': step.get('detail', ''),
        })
    results['step_timing'] = step_timing

    return jsonify(results)


# ====================================================
# 接口 10：下载文件
# GET /download/{task_id}/{file_key}
# ====================================================
@app.route('/tasks/<task_id>/gaussian-modes', methods=['GET'])
def get_gaussian_modes(task_id):
    """返回 Gaussian 振动模式向量数据（用于前端频率动画）"""
    if task_id not in gromacs.tasks:
        abort(404, '任务不存在')

    gaussian_results = gromacs.tasks[task_id].get('gaussian_results')
    if not gaussian_results:
        abort(404, 'Gaussian 结果不存在')

    normal_modes = gaussian_results.get('normal_modes')
    if not normal_modes:
        abort(404, '振动模式数据不存在')

    return jsonify({
        'atom_count': gaussian_results.get('atom_count', 0),
        'opt_xyz': gaussian_results.get('opt_xyz', ''),
        'modes': normal_modes
    })


@app.route('/download/<task_id>/<file_key>', methods=['GET'])
def download_file(task_id, file_key):
    """提供模拟输出文件的下载"""

    if task_id not in gromacs.tasks:
        abort(404, '任务不存在')

    work_dir = gromacs.tasks[task_id]['work_dir']

    file_mapping = {
        'xtc': ('md.xtc', 'trajectory.xtc'),
        'log': ('md.log', 'md.log'),
        'top': ('topol.top', 'topol.top'),
        'gro': ('md.gro', 'md.gro'),
        'edr': ('md.edr', 'md.edr'),
        'xvg': ('analysis.tar.gz', 'analysis.tar.gz'),
        'pdb': ('input.pdb', 'input.pdb'),
        'docked_pdb': ('docked_ligand.pdb', 'docked_ligand.pdb'),
        'docked_pdbqt': ('docked_ligand.pdbqt', 'docked_ligand.pdbqt'),
        'receptor_pdbqt': ('receptor.pdbqt', 'receptor.pdbqt'),
        'gaussian_log': ('ligand.log', 'ligand.log'),
        'gaussian_gjf': ('ligand.gjf', 'ligand.gjf'),
        'gaussian_opt_pdb': ('gaussian_opt.pdb', 'gaussian_opt.pdb'),
        'gaussian_chk': ('ligand.chk', 'ligand.chk'),
        'gaussian_homo_cube': ('homo.cube', 'homo.cube'),
        'gaussian_lumo_cube': ('lumo.cube', 'lumo.cube'),
        'gaussian_fchk': ('ligand.fchk', 'ligand.fchk'),
    }

    # 解析输入分子文件名（Gaussian 专用任务）
    task_info = gromacs.tasks[task_id]
    if task_info.get('task_type') == 'gaussian':
        params = task_info.get('params', {})
        mol_ext = params.get('molecule_ext', 'pdb')
        file_mapping['input_mol'] = (f'input.{mol_ext}', f'input.{mol_ext}')

    # 动态添加配体文件（如果存在）
    params = gromacs.tasks[task_id].get('params', {})
    ligand_path = params.get('ligand_path')
    if ligand_path and os.path.exists(ligand_path):
        ligand_filename = params.get('ligand_filename', 'ligand.xyz')
        _, ext = os.path.splitext(ligand_path)
        src_name = f'{task_id}_ligand{ext}'
        file_mapping['ligand'] = (os.path.join(os.path.dirname(gromacs.tasks[task_id]['work_dir']), src_name), ligand_filename)

    if file_key not in file_mapping:
        abort(404, '文件类型不存在')

    source_name, download_name = file_mapping[file_key]
    file_path = os.path.join(work_dir, source_name)

    if not os.path.exists(file_path):
        # 如果相对于 work_dir 找不到，尝试绝对路径
        for key, (src, _) in file_mapping.items():
            if key == file_key and os.path.isabs(src):
                file_path = src
                break
        else:
            abort(404, '文件尚未生成')

    return send_from_directory(
        work_dir,
        source_name,
        download_name=download_name,
        as_attachment=True
    )


# ====================================================
# 健康检查
# GET /health
# ====================================================
@app.route('/health', methods=['GET'])
def health():
    """服务器健康检查"""
    gmx_available = os.path.exists(gromacs.gmx) if os.path.sep in gromacs.gmx else bool(shutil.which(gromacs.gmx))
    return jsonify({
        'status': 'ok',
        'host': os.uname().nodename if hasattr(os, 'uname') else 'localhost',
        'gromacs': {
            'command': gromacs.gmx,
            'available': gmx_available,
        },
        'tasks_count': len(gromacs.tasks),
        'running_tasks': sum(1 for t in gromacs.tasks.values() if t['status'] == 'running')
    })


# ====================================================
# 程序入口
# ====================================================

# ====================================================
# 接口 11：AI 助手（多轮对话）
# POST /api/ai-assistant
# 参数：{ "session_id": "xxx", "message": "用户输入" }
# ====================================================
@app.route('/api/ai-assistant', methods=['POST'])
def ai_assistant():
    """AI 助手：多轮对话 + 自动执行模拟"""

    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': '请提供 message 字段'}), 400

    user_message = data['message'].strip()
    if not user_message:
        return jsonify({'error': '消息内容不能为空'}), 400

    # 用 client IP 或传入的 session_id 作为会话标识
    session_id = data.get('session_id') or request.remote_addr or 'default'

    result = ai_chat(session_id, user_message)

    if result.get('_action') == 'error':
        return jsonify({'error': result.get('error', 'AI 处理出错')}), 400

    action = result.get('action', 'chat')
    response_data = {
        'action': action,
        'reply': result.get('reply', ''),
    }

    # 如果是要运行模拟，附加任务信息
    if action == 'run_simulation':
        task_id = result.get('task_id', '')
        if task_id:
            task_params_path = os.path.join(app.config['TASKS_FOLDER'], task_id, 'params.json')
            import json
            if os.path.exists(task_params_path):
                with open(task_params_path, 'r') as f:
                    task_params = json.load(f)
                gromacs.create_task(task_id, task_params, task_type='md')
                from threading import Thread
                thread = Thread(target=gromacs.run_pipeline, args=(task_id,), daemon=True)
                thread.start()

        response_data['task_id'] = task_id
        response_data['steps'] = result.get('steps', [])
        response_data['params'] = result.get('params', {})
        response_data['pdb_info'] = result.get('pdb_info', {})
        response_data['ligand_info'] = result.get('ligand_info', {})

    return jsonify(response_data)


# ====================================================
# 程序入口
# ====================================================
if __name__ == '__main__':
    _gmx_ok = os.path.exists(gromacs.gmx) if os.path.sep in gromacs.gmx else bool(shutil.which(gromacs.gmx))
    print('=' * 60)
    print('  WebMD 模拟平台后端服务')
    print(f'  监听地址: http://0.0.0.0:5000')
    print(f'  GROMACS: {"✅ " + gromacs.gmx if _gmx_ok else "❌ 未找到"}')
    print('=' * 60)
    print()
    print('  可用接口:')
    print('    POST /upload-pdb      - 上传 PDB 文件')
    print('    POST /run-gromacs     - 启动 GROMACS 模拟')
    print('    GET  /tasks           - 任务列表')
    print('    GET  /tasks/<id>/status - 任务状态')
    print('    GET  /tasks/<id>/logs  - 任务日志')
    print('    POST /tasks/<id>/pause - 暂停任务')
    print('    POST /tasks/<id>/resume - 恢复任务')
    print('    POST /tasks/<id>/cancel - 取消任务')
    print('    GET  /tasks/<id>/results - 模拟结果')
    print('    GET  /download/<id>/<key> - 下载文件')
    print('    GET  /health          - 健康检查')
    print('    POST /api/ai-assistant - AI 助手（DeepSeek 智能配置）')
    print()

    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
