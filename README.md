# WebMD 模拟器

分子动力学模拟 Web 平台，集成 GROMACS 分子动力学、Gaussian 量子化学计算、AutoDock Vina 分子对接，支持 AI 智能助手。

## 部署

### 服务器要求

- Linux 系统（推荐 CentOS / Rocky Linux / Ubuntu）
- Python 3.10+
- GROMACS 2024+
- （可选）Gaussian 09/16
- （可选）AutoDock Vina
- （可选）Open Babel

### 部署步骤

```bash
# 1. 克隆项目到服务器
git clone <你的仓库地址>
cd web

# 2. 安装 Python 依赖
pip install -r backend/requirements.txt

# 3. 启动后端
cd backend
python app.py
```

服务默认监听 `0.0.0.0:5000`。

### 访问

浏览器打开 `http://服务器IP:5000` 即可使用。

### 修改 API 地址（重要）

前端 HTML 文件通过 `API_BASE` 变量连接后端。代码中已内置了自动识别逻辑：

```javascript
// 例如 submit.html 中的写法
const API_BASE = window.location.protocol === 'file:'
    ? 'http://192.168.16.130:5000'   // file:// 打开时，手动指定服务器地址
    : window.location.origin;         // 浏览器直接访问时，自动取当前域名
```

**什么情况下需要改？**

如果你在浏览器地址栏输入 `http://服务器IP:5000` 直接访问，**不需要改任何东西**，代码会自动用当前域名作为 API 地址。

如果你用浏览器**双击 HTML 文件**打开（地址栏是 `file:///C:/...`），则需要把 `http://192.168.16.130:5000` 改成你实际的服务器 IP。

**需要修改的页面**（共 4 个，搜索 `API_BASE` 即可找到）：

| 文件 | 位置 |
|------|------|
| `submit.html` | 第 395 行 |
| `run.html` | 第 331 行 |
| `results.html` | 第 752 行 |
| `gaussian.html` | 第 304 行 |
| `ai_assistant.html` | 第 258 行 |

## 项目结构

```
web/
├── .gitignore
├── index.html             # 首页
├── submit.html            # 提交模拟任务
├── run.html               # 运行监控
├── results.html           # 结果展示
├── gaussian.html          # 量子化学计算
├── ai_assistant.html      # AI 助手
└── backend/
    ├── app.py             # Flask API 服务
    ├── gromacs_runner.py  # GROMACS 任务管理器
    ├── analysis.py        # 结果分析模块
    ├── mdp_templates.py   # MDP 参数模板
    ├── ai_assistant.py    # AI 助手（DeepSeek API）
    ├── ai_config.py       # AI 配置
    ├── requirements.txt   # Python 依赖
    ├── tasks/             # 任务数据（自动生成）
    └── uploads/           # 上传文件（自动生成）
```

## 功能说明

| 页面 | 功能 |
|------|------|
| 首页 | 平台介绍及导航 |
| 提交任务 | 上传 PDB 文件，配置力场/水模型/温度等参数，启动 MD 模拟 |
| 量子化学 | 独立 Gaussian 计算（单点能/构型优化/频率分析） |
| AI 助手 | 自然语言描述需求，自动配置并启动模拟 |
| 运行监控 | 实时查看任务进度和日志 |
| 结果展示 | RMSD/能量/氢键/RDF/3D 结构/对接结果等可视化分析 |

### 支持的力场

- AMBER99SB-ILDN（默认）
- AMBER19SB (ff19SB)
- CHARMM36
- OPLS-AA
- GROMOS 54A7

## 技术栈

- **前端**: HTML5, Bootstrap 5.3, ECharts 5.4.3, 3Dmol.js
- **后端**: Python Flask 3.0+
- **计算引擎**: GROMACS, Gaussian, AutoDock Vina
- **AI**: DeepSeek API (deepseek-v4-flash)
