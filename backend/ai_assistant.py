import os
import re
import json
import requests
from datetime import datetime

# ====================================================
# DeepSeek 配置（优先级：环境变量 > ai_config.py > 默认值）
# ====================================================
try:
    from ai_config import DEEPSEEK_API_KEY as _CONFIG_KEY, DEEPSEEK_API_URL as _CONFIG_URL, DEEPSEEK_MODEL as _CONFIG_MODEL
except (ImportError, ModuleNotFoundError):
    _CONFIG_KEY = ''
    _CONFIG_URL = 'https://api.deepseek.com/v1/chat/completions'
    _CONFIG_MODEL = 'deepseek-v4-flash'

DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', _CONFIG_KEY)
DEEPSEEK_API_URL = os.environ.get('DEEPSEEK_API_URL', _CONFIG_URL)
DEEPSEEK_MODEL = os.environ.get('DEEPSEEK_MODEL', _CONFIG_MODEL)

# 内存会话存储 { session_id: [messages] }
sessions = {}

# ====================================================
# 系统提示词：AI 助手的角色定义
# ====================================================
SYSTEM_PROMPT = """你是一个分子动力学模拟专家助手。你有两种回复模式：

## 模式1: chat（聊天）
当用户没有明确要求模拟，或只是提问、打招呼时：
{
  "action": "chat",
  "reply": "你的回复内容"
}

## 模式2: run_simulation（运行模拟）
当用户明确要求模拟某个蛋白，或你确认了 PDB 和配体信息后：
{
  "action": "run_simulation",
  "reply": "告诉用户你要开始做什么",
  "pdb_id": "四位PDB代码，如1AKI",
  "ligand_name": "配体名称或CID（没有则null）",
  "parameters": {
    "force_field": "amber99sb-ildn",
    "water_model": "tip3p",
    "temperature": 300,
    "simulation_time_ps": 1000,
    "pressure": 1.0,
    "ion_type": "na-cl",
    "ion_concentration": 0.15,
    "ph": 7.0,
    "enable_docking": false,
    "enable_gaussian": false
  },
  "reasoning": "参数选择理由"
}

规则：
1. 如果用户说"你好"、"帮我看看"等模糊请求 → 用 chat 模式回复，询问具体要求
2. 如果用户给了 PDB ID（如1AKI）或明确说要模拟 → 用 run_simulation 模式
3. 如果 PDB ID 不全（比如只有数字没有字母），用 chat 模式请用户补充
4. 用户可能用中文名（如"溶菌酶"）→ 用 chat 模式告诉用户需要 PDB ID
5. 始终用中文回复
6. 用力场：AMBER99SB-ILDN（默认）、AMBER19SB、CHARMM36、OPLS-AA、GROMOS54A7；水模型 TIP3P（默认）、SPCE、TIP4P；温度 300K；pH 7.0
7. 如果用户提到配体，enable_docking 设为 true
"""


def call_deepseek(messages):
    """调用 DeepSeek API，传入完整消息历史"""
    if not DEEPSEEK_API_KEY:
        return {'error': 'DeepSeek API 密钥未配置。', '_action': 'error'}

    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers={
                'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={
                'model': DEEPSEEK_MODEL,
                'messages': messages,
                'temperature': 0.3,
                'max_tokens': 4096,
            },
            timeout=60
        )

        if response.status_code != 200:
            return {
                'error': f'DeepSeek API 返回错误 (HTTP {response.status_code})',
                '_action': 'error'
            }

        data = response.json()
        content = data['choices'][0]['message']['content'].strip()
        if not content:
            return {'error': 'DeepSeek 返回了空内容', '_action': 'error'}

        # 多方法提取 JSON（兼容有无代码块、有无额外文本）
        json_str = None

        # 方式1: 从代码块提取
        m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if m:
            json_str = m.group(1).strip()

        # 方式2: 找第一个 { 到最后一个 }
        if not json_str:
            s = content.find('{')
            e = content.rfind('}')
            if s != -1 and e != -1 and e > s:
                json_str = content[s:e+1]

        # 方式3: 整个内容
        if not json_str:
            json_str = content

        if not json_str or not json_str.strip():
            return {'error': 'DeepSeek 返回的内容无法解析为 JSON', '_action': 'error', '_raw_reply': content[:500]}

        parsed = json.loads(json_str)
        parsed['_raw_reply'] = content

        if 'action' not in parsed:
            parsed['action'] = 'chat'
        if 'reply' not in parsed:
            parsed['reply'] = parsed.get('reasoning', '已收到你的请求。')

        return parsed

    except json.JSONDecodeError as e:
        return {'error': f'JSON 解析失败: {str(e)}', '_action': 'error', '_raw_reply': content if 'content' in locals() else ''}
    except requests.exceptions.Timeout:
        return {'error': 'DeepSeek API 请求超时', '_action': 'error'}
    except Exception as e:
        return {'error': f'请求出错: {str(e)}', '_action': 'error'}


# ====================================================
# 会话管理
# ====================================================
def get_session(session_id):
    """获取或创建会话"""
    if session_id not in sessions:
        sessions[session_id] = [
            {'role': 'system', 'content': SYSTEM_PROMPT}
        ]
    return sessions[session_id]


def chat(session_id, user_message):
    """
    多轮对话入口
    返回: { action, reply, task_id?, steps?, pdb_info?, ligand_info?, params? }
    """
    # 1. 获取会话历史
    messages = get_session(session_id)

    # 2. 追加用户消息
    messages.append({'role': 'user', 'content': user_message})

    # 3. 调用 DeepSeek
    result = call_deepseek(messages)

    if '_action' in result and result['_action'] == 'error':
        return result

    action = result.get('action', 'chat')
    reply = result.get('reply', '')

    # 4. 追加 AI 回复到历史
    messages.append({'role': 'assistant', 'content': json.dumps(result, ensure_ascii=False)})

    # 5. 如果是要运行模拟
    if action == 'run_simulation':
        pdb_id = result.get('pdb_id', '').strip().upper()
        if not pdb_id or not re.match(r'^[A-Z0-9]{4}$', pdb_id):
            return {
                'action': 'chat',
                'reply': f'你提到的 "{result.get("pdb_id", "")}" 看起来不是有效的 PDB ID（需要4位字母数字代码，如 1AKI）。请提供正确的 PDB ID。'
            }

        # 开始下载并准备任务
        steps = [{'type': '思考', 'content': f'准备模拟 **{pdb_id}**...'}]
        steps.append({'type': '分析', 'content': result.get('reasoning', '')[:200]})

        upload_folder = os.path.join(os.path.dirname(__file__), 'MoleculeDownload')
        tasks_folder = os.path.join(os.path.dirname(__file__), 'tasks')
        os.makedirs(upload_folder, exist_ok=True)
        os.makedirs(tasks_folder, exist_ok=True)

        # 下载 PDB
        steps.append({'type': '下载', 'content': f'正在从 RCSB PDB 下载 **{pdb_id}**...'})
        pdb_path, pdb_result = download_pdb(pdb_id, upload_folder)
        if isinstance(pdb_result, str):
            return {'action': 'chat', 'reply': f'下载失败: {pdb_result}。请检查 PDB ID 是否正确。'}
        pdb_result['file_path'] = pdb_path
        steps.append({'type': '成功', 'content': f'PDB 下载完成: {pdb_result.get("title", pdb_id)}'})

        # 下载配体
        ligand_downloaded = None
        ligand_name = result.get('ligand_name')
        if ligand_name and ligand_name != 'null' and ligand_name.lower() != 'none':
            steps.append({'type': '下载', 'content': f'正在从 PubChem 下载配体 **{ligand_name}**...'})
            lig_path, lig_result = download_ligand(ligand_name, upload_folder)
            if isinstance(lig_result, str):
                steps.append({'type': '警告', 'content': f'配体 {ligand_name} 下载失败'})
            else:
                lig_result['file_path'] = lig_path
                ligand_downloaded = lig_result
                steps.append({'type': '成功', 'content': f'配体 {ligand_name} 下载完成'})

        # 创建任务
        steps.append({'type': '配置', 'content': '正在创建模拟任务...'})
        params = result.get('parameters', {})
        task_id, task_params = create_task_from_ai(pdb_result, ligand_downloaded, params, tasks_folder)
        steps.append({'type': '完成', 'content': f'任务 **{task_id}** 已创建，正在启动...'})

        return {
            'action': 'run_simulation',
            'reply': reply,
            'task_id': task_id,
            'steps': steps,
            'params': params,
            'pdb_info': pdb_result,
            'ligand_info': ligand_downloaded,
        }

    # 6. 普通聊天回复
    return {
        'action': 'chat',
        'reply': reply,
    }


def download_pdb(pdb_id, save_dir):
    """从 RCSB PDB 下载蛋白质结构文件"""
    pdb_id = pdb_id.strip().upper()
    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    save_path = os.path.join(save_dir, f'{pdb_id}.pdb')
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return None, f'PDB 下载失败 (HTTP {resp.status_code})'
        content = resp.text
        if 'HEADER' not in content and 'ATOM' not in content:
            return None, f'不是有效的 PDB 格式'
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(content)
        title = ''
        for line in content.split('\n'):
            if line.startswith('TITLE') and not title:
                title = line[10:].strip()
        return save_path, {
            'message': 'PDB 下载成功',
            'pdb_id': pdb_id,
            'title': title or pdb_id,
            'file_size': f'{len(content) / 1024:.1f} KB',
            'atom_count': len([l for l in content.split('\n') if l.startswith(('ATOM', 'HETATM'))]),
        }
    except Exception as e:
        return None, f'下载出错: {str(e)}'


def download_ligand(ligand_name, save_dir):
    """从 PubChem 下载配体"""
    try:
        if ligand_name.isdigit():
            url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{ligand_name}/SDF'
        else:
            search_url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{ligand_name}/cids/JSON'
            search_resp = requests.get(search_url, timeout=15)
            if search_resp.status_code == 200:
                cids = search_resp.json().get('IdentifierList', {}).get('CID', [])
                if not cids:
                    return None, '未找到配体'
                ligand_name = str(cids[0])
                url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{ligand_name}/SDF'
            else:
                return None, '搜索配体失败'
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return None, f'配体下载失败 (HTTP {resp.status_code})'
        content = resp.text
        save_path = os.path.join(save_dir, f'{ligand_name}.sdf')
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return save_path, {'message': '配体下载成功', 'name': ligand_name, 'file_size': f'{len(content) / 1024:.1f} KB'}
    except Exception as e:
        return None, f'下载出错: {str(e)}'


def create_task_from_ai(pdb_result, ligand_result, params, tasks_folder):
    """创建模拟任务"""
    import json as json_module
    from datetime import datetime

    pdb_id = pdb_result.get('pdb_id', 'protein')
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', pdb_id)[:20].strip('_') or 'protein'
    task_id = f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    task_params = {
        'pdb_path': pdb_result.get('file_path', ''),
        'pdb_filename': f'{pdb_id}.pdb',
        'ligand_path': ligand_result.get('file_path', '') if ligand_result else None,
        'ligand_filename': f"{ligand_result.get('name', 'ligand')}.sdf" if ligand_result else None,
        'force_field': params.get('force_field', 'amber99sb-ildn'),
        'water_model': params.get('water_model', 'tip3p'),
        'ion_type': params.get('ion_type', 'na-cl'),
        'ion_concentration': params.get('ion_concentration', 0.15),
        'temperature': params.get('temperature', 300),
        'simulation_time': params.get('simulation_time_ps', 1000),
        'time_step': 2.0,
        'pressure': params.get('pressure', 1.0),
        'description': pdb_result.get('title', '') or pdb_id,
        'solvate': True,
        'docking_enabled': params.get('enable_docking', False),
        'center_x': 0, 'center_y': 0, 'center_z': 0,
        'size_x': 20, 'size_y': 20, 'size_z': 20,
        'exhaustiveness': 8,
        'gaussian_enabled': params.get('enable_gaussian', False),
        'gaussian_calc_type': 'Opt', 'gaussian_method': 'B3LYP',
        'gaussian_basis': '6-311++G(d,p)', 'gaussian_charge': 0,
        'gaussian_multiplicity': 1, 'gaussian_nproc': 4, 'gaussian_mem': '4GB',
    }

    task_dir = os.path.join(tasks_folder, task_id)
    os.makedirs(task_dir, exist_ok=True)
    with open(os.path.join(task_dir, 'params.json'), 'w') as f:
        json_module.dump(task_params, f, indent=2)

    return task_id, task_params
