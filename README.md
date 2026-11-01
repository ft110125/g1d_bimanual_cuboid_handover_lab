# G1-D Bimanual Cuboid Handover — Teleoperation & Imitation Learning

Unitree G1-D（Revo2ハンド付き）を**両腕**で操作し、直方体を右手から左手へ手渡すタスクです。  
Meta Quest 3のハンドトラッキングでテレオペレーション → デモ収集 → Isaac Lab Mimic → robomimic模倣学習パイプラインを実現します。

> **注意**: 本プロジェクトは `g1d_bimanual_xr_teleop_lab` を参考に新規作成した**独立プロジェクト**です。  
> 参考プロジェクトは一切変更していません。実行時依存もありません。

---

## タスク概要

1. テーブル右側に置かれた直方体を**右手**で把持
2. 右手で持ち上げる
3. 中央の手渡し位置まで運ぶ
4. **右手→左手**へ手渡す
5. 右手を離す
6. 左手でテーブル左側の目標位置まで運ぶ
7. 左手でテーブル上に置く
8. 左手を離してタスク完了

### 直方体の寸法

| 軸 | 実寸 | Isaac Lab (m) |
|----|------|---------------|
| X (奥行き) | 150 mm | 0.150 |
| Y (幅) | 117 mm | 0.117 |
| Z (高さ) | 60 mm | 0.060 |

### 座標系と配置

| パラメータ | 値 | 備考 |
|-----------|-----|------|
| テーブル上面Z | 0.85 m | |
| 直方体初期位置 | (0.75, -0.22, 0.88) | 右側、テーブル上 |
| 手渡し位置 | (0.70, 0.00, 1.02) | 中央、テーブルより上 |
| 配置目標位置 | (0.75, 0.22, 0.88) | 左側、テーブル上 |
| 直方体質量 | 0.5 kg | |

座標系: X=前方、Y=左方、Z=上方（Isaac Lab標準ロボット座標）

---

## 登録環境ID

| ID | 用途 | アクション次元 |
|----|------|-------------|
| `G1D-Bimanual-CuboidHandover-IK-Rel-v0` | テレオペ・学習 | 24 |
| `G1D-Bimanual-CuboidHandover-IK-Rel-PLAY-v0` | 評価・可視化 | 24 |
| `G1D-Bimanual-CuboidHandover-IK-Rel-Mimic-v0` | Mimicデモ増殖 | 24 |
| `G1D-Bimanual-CuboidHandover-Pink-Abs-v0` | Pink IK テレオペ | 26 |
| `G1D-Bimanual-CuboidHandover-Pink-Abs-PLAY-v0` | Pink IK 評価 | 26 |

### アクション構造

**IK-Rel (24次元)**:
```
[0:6]   右腕 EEF (arm_action) — Diff IK relative
[6:12]  右手指 (gripper_action) — Revo2 6ch
[12:18] 左腕 EEF (left_arm_action) — Diff IK relative
[18:24] 左手指 (left_gripper_action) — Revo2 6ch
```

**Pink-Abs (26次元)**:
```
[0:14]  両腕 (arm_action) — left wrist 7 + right wrist 7
[14:20] 右手指 (gripper_action) — Revo2 6ch
[20:26] 左手指 (left_gripper_action) — Revo2 6ch
```

> **重要**: IK-RelとPinkではアクション次元が異なります。Mimic環境はIK-Rel (24D)専用です。

---

## ディレクトリ構成

```
g1d_bimanual_cuboid_handover_lab/
├── README.md
├── .gitignore
├── .env.teleop.example
├── start_teleop_xr.sh                    # XRテレオペ起動
├── scripts/
│   ├── run_isaaclab_tool.py              # 外部env登録付きスクリプト実行
│   └── list_envs.py                      # 登録済み環境一覧
├── datasets/
│   └── g1d_bimanual_cuboid_handover/     # デモデータ保存先
│       ├── dataset.hdf5                  # 収集デモ
│       ├── annotated_dataset.hdf5        # Mimic注釈済み
│       └── generated_dataset.hdf5        # Mimic生成済み
└── source/
    └── g1d_bimanual_cuboid_handover/
        ├── pyproject.toml
        └── g1d_bimanual_cuboid_handover/
            ├── __init__.py
            ├── assets/
            │   ├── __init__.py
            │   └── g1d.py                              # G1-D ArticulationCfg
            └── tasks/
                ├── __init__.py
                └── manager_based/
                    ├── __init__.py
                    └── g1d_bimanual/
                        ├── __init__.py                          # Gym環境登録
                        ├── g1d_bimanual_env_cfg.py              # 環境Config + 定数
                        ├── g1d_bimanual_mimic_env.py            # Mimic wrapper + phase detection
                        ├── g1d_bimanual_mimic_env_cfg.py        # Mimic subtask config
                        ├── g1d_se3_rel_retargeter.py            # 腕 retargeter (relative)
                        ├── g1d_se3_abspos_relrot_retargeter.py  # 腕 retargeter (abs+rel)
                        ├── g1d_abspos_relrot_ik_action.py       # AbsPosRelRot IK ActionTerm
                        ├── g1d_bimanual_abs_wrist_retargeter.py # Pink wrist retargeter
                        ├── g1d_pink_upper_body_action.py        # Pink IK ActionTerm
                        ├── g1d_pink_controller_cfg.py           # Pink controller builder
                        ├── g1d_teleop_tuning.py                 # 環境変数→tuning
                        ├── g1d_revo2_hand_math.py               # 手形状計算
                        ├── g1d_revo2_hand_retargeter.py         # Revo2 6ch retargeter
                        └── agents/
                            ├── __init__.py
                            ├── rsl_rl_ppo_cfg.py
                            └── robomimic/
                                └── bc_handover.json
```

---

## 必要な環境

- Isaac Sim (Isaac Lab 2.3.2 が動作する対応バージョン)
- Isaac Lab 2.3.2
- Python 3.10+
- CUDA対応GPU
- Meta Quest 3 (テレオペ時)
- CloudXR Runtime (テレオペ時)

### バージョン確認

```bash
cat /workspace/isaaclab/VERSION          # Isaac Lab バージョン
python -c "import isaacsim; print(isaacsim.__version__)"  # Isaac Sim バージョン
```

---

## インストール

```bash
cd /workspace/my_isaac_project/g1d_bimanual_cuboid_handover_lab
python -m pip install -e source/g1d_bimanual_cuboid_handover
```

### 環境登録確認

```bash
cd /workspace/my_isaac_project/g1d_bimanual_cuboid_handover_lab
python scripts/list_envs.py
```

---

## VR/XR テレオペレーション起動

### IK-Rel モード（推奨・デモ収集用）

```bash
cd /workspace/my_isaac_project/g1d_bimanual_cuboid_handover_lab
bash start_teleop_xr.sh
```

### Pink モード

```bash
cd /workspace/my_isaac_project/g1d_bimanual_cuboid_handover_lab
G1D_TELEOP_TASK=G1D-Bimanual-CuboidHandover-Pink-Abs-v0 bash start_teleop_xr.sh
```

---

## デモ収集

```bash
cd /workspace/my_isaac_project/g1d_bimanual_cuboid_handover_lab
python scripts/run_isaaclab_tool.py \
    /workspace/isaaclab/scripts/environments/teleoperation/teleop_se3_agent.py \
    --task G1D-Bimanual-CuboidHandover-IK-Rel-v0 \
    --teleop_device handtracking \
    --num_envs 1 \
    --device cuda:0 \
    --save_dataset datasets/g1d_bimanual_cuboid_handover/dataset.hdf5
```

---

## デモ再生

```bash
cd /workspace/my_isaac_project/g1d_bimanual_cuboid_handover_lab
python scripts/run_isaaclab_tool.py \
    /workspace/isaaclab/scripts/tools/replay_demonstrations.py \
    --task G1D-Bimanual-CuboidHandover-IK-Rel-PLAY-v0 \
    --dataset datasets/g1d_bimanual_cuboid_handover/dataset.hdf5 \
    --device cuda:0
```

---

## Mimic — 自動注釈

```bash
cd /workspace/my_isaac_project/g1d_bimanual_cuboid_handover_lab
python scripts/run_isaaclab_tool.py \
    /workspace/isaaclab/scripts/tools/auto_annotate_demos.py \
    --task G1D-Bimanual-CuboidHandover-IK-Rel-Mimic-v0 \
    --input datasets/g1d_bimanual_cuboid_handover/dataset.hdf5 \
    --output datasets/g1d_bimanual_cuboid_handover/annotated_dataset.hdf5 \
    --device cuda:0
```

---

## Mimic — データ生成

```bash
cd /workspace/my_isaac_project/g1d_bimanual_cuboid_handover_lab
python scripts/run_isaaclab_tool.py \
    /workspace/isaaclab/scripts/tools/generate_demos.py \
    --task G1D-Bimanual-CuboidHandover-IK-Rel-Mimic-v0 \
    --input datasets/g1d_bimanual_cuboid_handover/annotated_dataset.hdf5 \
    --output datasets/g1d_bimanual_cuboid_handover/generated_dataset.hdf5 \
    --num_envs 16 \
    --num_demos 100 \
    --device cuda:0
```

---

## robomimic 学習 (BC)

```bash
cd /workspace/my_isaac_project/g1d_bimanual_cuboid_handover_lab
python scripts/run_isaaclab_tool.py \
    /workspace/isaaclab/scripts/tools/train_robomimic.py \
    --task G1D-Bimanual-CuboidHandover-IK-Rel-v0 \
    --dataset datasets/g1d_bimanual_cuboid_handover/generated_dataset.hdf5 \
    --device cuda:0
```

> **注意**: ハンドオーバーは時間依存性が強いタスクです。  
> `seq_length=1` かつ RNN無効の標準BC設定では性能が不足する可能性があります。  
> 必要に応じてRNN (`"rnn": {"enabled": true}`)、または観測履歴の導入を検討してください。  
> 最初は基本パイプラインの動作確認を優先してください。

---

## 学習済みポリシー再生

```bash
cd /workspace/my_isaac_project/g1d_bimanual_cuboid_handover_lab
python scripts/run_isaaclab_tool.py \
    /workspace/isaaclab/scripts/tools/play_robomimic.py \
    --task G1D-Bimanual-CuboidHandover-IK-Rel-PLAY-v0 \
    --checkpoint <path_to_model.pth> \
    --device cuda:0
```

---

## 成功判定

最終success（ステートレス、デモ記録用）:

| 条件 | 閾値 |
|------|------|
| XY位置誤差 | < 0.05 m |
| Z位置誤差 | < 0.025 m |
| 線速度 | < 0.08 m/s |
| 角速度 | < 0.50 rad/s |
| 右手クリアランス | > 0.14 m |
| 左手クリアランス | > 0.14 m |

---

## サブタスク信号 (Mimic)

| 信号名 | 意味 |
|--------|------|
| `right_grasped` | 右手が直方体を把持して持ち上げた |
| `handover_ready` | 直方体が手渡し位置に到達 |
| `left_grasped` | 左手が直方体を把持 |
| `right_released` | 右手が直方体から離れた |
| `placed` | 直方体が目標位置に配置 |
| `final_success` | 左手も離れてタスク完了 |

### フェーズ判定方法

**本バージョンでは接触センサーを使用していません。**

フェーズ判定は以下の近似判定で行います:
- EEFと物体の距離
- 物体の持ち上がり高さ (テーブル上 +5cm)
- 物体の手渡し領域への到達 (半径 8cm)
- 物体の速度
- フェーズ順序 (ラッチ形式)

将来的には `ContactSensorCfg` を左右指リンクに追加して精度を向上可能。

---

## Mimic左右同期設定

手渡しの左右タイミングを保つため、以下の設定を使用:

```python
generation_select_src_per_subtask = False  # 同じデモソースを使用
```

手渡しフェーズのアクションノイズは 0.005 に抑制。

---

## 観測キー

| キー | 次元 | 説明 |
|------|------|------|
| `joint_pos` | variable | 関節位置 (相対) |
| `joint_vel` | variable | 関節速度 (相対) |
| `object_position` | 3 | 直方体位置 |
| `object_orientation` | 4 | 直方体姿勢 (quaternion) |
| `object_linear_velocity` | 3 | 直方体線速度 |
| `object_angular_velocity` | 3 | 直方体角速度 |
| `target_object_position` | 3 | 配置目標位置 |
| `handover_position` | 3 | 手渡し位置 |
| `actions` | 24 | 前ステップアクション |
| `right_eef_pos` | 3 | 右EEF位置 |
| `right_eef_quat` | 4 | 右EEF姿勢 |
| `left_eef_pos` | 3 | 左EEF位置 |
| `left_eef_quat` | 4 | 左EEF姿勢 |

---

## エピソード時間

- **35秒** (参考プロジェクトの15秒より長い)
- ハンドオーバーは複数フェーズで時間を要するため

---

## 既知の制約

1. **接触センサー未使用** — フェーズ判定はEEF距離+高さの近似
2. **BC (seq_length=1)** — 時間依存タスクには不十分な可能性
3. **初期位置固定** — ランダム化なし（初期開発版）
4. **GPU/VRが必要** — headlessでの完全動作確認にはGPU必須
5. **Pink版はMimic非対応** — Mimicはアクション次元24 (IK-Rel) 前提

---

## トラブルシューティング

### ImportError: g1d_bimanual_cuboid_handover
```bash
cd /workspace/my_isaac_project/g1d_bimanual_cuboid_handover_lab
python -m pip install -e source/g1d_bimanual_cuboid_handover
```

### 環境が見つからない
```bash
python scripts/list_envs.py
```

### OpenXR/CloudXR接続エラー
- `XR_RUNTIME_JSON` が正しいパスを指しているか確認
- CloudXR serviceが起動しているか確認: `ps aux | grep start_service`

### 直方体がテーブルから落ちる
- 摩擦係数を確認 (static=1.0, dynamic=0.8)
- `solver_position_iteration_count` を増やす

### Mimicデータ生成が失敗する
- 十分な数のデモ (10+) を収集してから実行
- `generation_num_trials` を増やす
- 手渡しデモで全フェーズが成立しているか確認

### 「右手で直接左側に置く」動作になる
- Mimicのサブタスク信号で `right_grasped` → `handover_ready` → `left_grasped` の順序を強制
- デモ収集時に正しい手渡し手順を実行すること
