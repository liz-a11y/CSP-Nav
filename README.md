# CSP-MRNav source release

这是一个用于公开到 GitHub 的精简源码目录。它保留 CSP/LMTE/CBF、多机器人环境和 MAPPO 的核心实现，同时暂不公开训练参数配置。

## 当前公开范围

- `crowd_sim/social/`：LMTE、CSP 指标、奖励、动作过滤和复合社会 CBF。
- `crowd_sim/wrappers/`：CSP/LMTE 与多机器人环境的集成层。
- `crowd_sim/envs/` 与 `crowd_sim/multi_robot_core.py`：多机器人环境和状态契约。
- `training/algo/` 与 `training/networks/`：MAPPO 优化器、Actor/Critic、存储和向量环境。
- `crowd_nav/policy/`：环境运行所需的基础策略接口。
- `train_mappo.py` 与 `training/test_mappo.py`：训练和评估入口；配置改为运行时注入。
- `tests/`：不依赖私有训练参数的核心单元测试。

## 暂不公开的内容

- `crowd_nav/configs/config.py`、`config_mappo.py` 及其他训练配置。
- checkpoint、权重、数据、日志、运行目录和配置快照。
- 论文、结果图、实验导出、内部实施记录和真实机器人部署说明。
- AVOCADO、CEMRRL、HeR-DRL、RL-RVO 等第三方基线源码。
- PyBullet/TurtleBot 模型和纹理；其再分发许可确认前不放入公开目录。

因此，当前版本可以审阅和测试核心方法，但在配置与仿真资源补齐前，不能复现端到端训练结果。若以后公开 checkpoint，还必须同时公开与其匹配的观测、动作和网络结构配置，否则模型权重无法可靠加载。

## 私有配置的本地使用

把私有的 `config.py` 和 `config_mappo.py` 放到 `crowd_nav/configs/`。这些文件已被 `.gitignore` 排除，不要使用 `git add -f` 强制提交。

配置模块必须提供 `Config` 类。训练入口在运行时加载它，因此即使公开目录中没有配置，`python train_mappo.py --help` 仍可使用。

```powershell
python train_mappo.py --config-module crowd_nav.configs.config_mappo --cpu
```

评估时，`training/test_mappo.py` 只从 `<model_dir>/configs/` 读取与 checkpoint 配套的私有配置快照：

```powershell
python -m training.test_mappo --checkpoint <model_dir>/checkpoints/<checkpoint>.pt --cpu
```

## 环境安装

已验证的原始开发环境为 Python 3.8。先安装 Python 依赖：

```powershell
python -m pip install -r requirements.txt
```

此外需要分别安装 [OpenAI Baselines](https://github.com/openai/baselines) 和 [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2)。它们没有复制到本仓库。

## 测试

```powershell
python -m pytest -q
```

这里的测试覆盖公开核心模块，不代表论文基准复现，也不验证未公开配置下的训练性能。

## 发布前检查

1. 在 `csp/` 中新建独立 Git 仓库，不继承当前工作目录的历史。
2. 确认 `git status --short --ignored` 中配置、权重、数据和日志均被忽略。
3. 对待提交文件运行 secret scanner，并人工检查绝对路径、IP、用户名和密码。
4. 核实 PyBullet/TurtleBot 资源的再分发许可后，再按源码引用的原目录结构补充资源。
5. 保留 `LICENSE` 和 `THIRD_PARTY_NOTICES.md` 中的上游署名。

## 许可

本目录是 HEIGHT/CrowdNav_HEIGHT 的派生源码，继续保留上游 MIT 许可声明。第三方依赖和未随仓库分发的资源遵循各自许可，详见 `THIRD_PARTY_NOTICES.md`。
