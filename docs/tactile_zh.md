# 可选触觉仿真

[安装](tactile-installation.md) · [三任务完整命令](../tests/tactile/integration/README.md) · [English](tactile.md)

继续使用原生入口 `scripts/robodojo.sh`。选择 `--env-cfg arx_x5_pad_tactile`
启用当前 CAD 胶垫方案；原版 `arx_x5` 默认不开启触觉。

当前配置沿用双 X5 原生任务布局，**只有左臂夹爪的两个指尖各装一个 1 mm 胶垫**，
右臂保持原版。`Assets/Robots/x5_pad` 仍引用 `Assets/Robots/x5`，两者都要保留。

先启动策略服务器，再运行原生 `client`；充电器、管子、硬币的配套命令统一写在
[三任务说明](../tests/tactile/integration/README.md)。`--gui` 显示窗口，省略则无窗口运行。
`--eval-num` 控制回合数，`--num-envs` 控制并行环境数；当前触觉仅支持单环境。

传感器已经接入场景初始化、物理步进、观测、reset 和 close。
`obs["tactile"]["left" / "right"]` 提供当前 RGB、参考 RGB、有效性与帧时间；
这里的左右指同一夹爪的两个指尖。

运行会生成普通相机视频及三相机、两触觉的 `*_visuotactile_*.mp4`，不再自动保存 NPZ。
原生结果、服务器轨迹等中间文件放入 `.cache`，需要交付的视频和结果 JSON 再复制到 `outputs`。

当前接受的效果使用 UIPC 接触/FEM 物理及 `optical_input: object`：虚拟物体深度经
GelSight Mini 的 Taxim 响应得到 RGB。可选的 `deformed_surface` 光学仍保留，但当前配置未启用。
三个 demo 使用已知场景几何生成参考抓取轨迹，用于观察接触、保持和松开，不能作为模型任务成功率。

已有资产无需重新导入。只有更新 CAD 时，才使用
`scripts/internal/import_x5_pad.py --source-dir <解压目录>`；保留原始 CAD 和生成的来源清单。
