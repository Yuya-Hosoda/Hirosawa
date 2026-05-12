# 仕様書：充電制約付きマルチエージェント経路計画シミュレータ（最終版）

## 文書情報

| 項目 | 内容 |
|---|---|
| 対象論文 | 「Mobile Moduleにおける充電制約付きマルチエージェント経路計画手法の提案」（廣澤考冶, 2025年度） |
| 文書目的 | 論文の提案手法をPythonで再現するための詳細仕様 |
| 対象言語 | Python 3.11 |
| 主要ライブラリ | NumPy, Matplotlib（論文準拠） |
| v3→最終版 改訂 | 回転コスト修正（1レイヤ1tick）、離散化/実バッテリー使い分け明記、Conflict両位置保持、charging goalのLow-Levelゴール条件追加、ECBS制約生成規則追加 |

### 再現モードの定義

本仕様の再現には2段階のモードを設ける。

```
モード1: 論文記述準拠モード
  本文で明示された式・制約・パラメータのみを用いる。
  CS_SELECTION_PARAMS は初期値で固定。
  Scenario1 座標は Fig.5.1 からの推定値をそのまま使用。
  結果が Table 5.7/5.8 と乖離してもパラメータ調整は行わない。
  検証基準: 改善方向の一致・安全性・定性的傾向の再現。

モード2: Table較正モード
  本文に数値がない CS_SELECTION_PARAMS と Scenario1 座標を較正し、
  Table 5.7/5.8 の目標値に近づける。
  検証基準: 平均完了タスク数が目標値の ±20% 以内（推奨 ±5%）。
```

---

## 1. システム全体構成

### 1.1 目的

天井面移動ロボットMoMo5の複数台を対象に、バッテリー残量を監視しながら充電ステーション（CS）への立ち寄りを経路に組み込み、エージェント間の衝突を回避するLifelong MAPF経路計画シミュレータを構築する。

### 1.2 システム構成図

```
┌─────────────────────────────────────────────────┐
│                 SimulationEngine                │
│  ┌───────────┐  ┌────────────┐  ┌────────────┐  │
│  │ GridMap    │  │ AgentMgr   │  │ TaskMgr    │  │
│  │ (環境)    │  │ (状態管理) │  │ (タスク列) │  │
│  └───────────┘  └────────────┘  └────────────┘  │
│  ┌───────────────────────────────────────────┐  │
│  │          ChargingScheduler                │  │
│  │  ┌─────────────┐  ┌───────────────────┐  │  │
│  │  │ CSSelector   │  │ EmergencySystem   │  │  │
│  │  │ (CS選択)     │  │ (緊急介入)        │  │  │
│  │  └─────────────┘  └───────────────────┘  │  │
│  └───────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────┐  │
│  │              PathPlanner                  │  │
│  │  ┌──────────────┐  ┌──────────────────┐  │  │
│  │  │ ECBS         │  │ LowLevelPlanner  │  │  │
│  │  │ (高レベル)   │  │ (Weighted A*)    │  │  │
│  │  └──────────────┘  └──────────────────┘  │  │
│  │  ┌──────────────┐  ┌──────────────────┐  │  │
│  │  │ Conflict     │  │ Heuristic        │  │  │
│  │  │ Detector     │  │ (h関数)          │  │  │
│  │  └──────────────┘  └──────────────────┘  │  │
│  └───────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────┐  │
│  │           StatisticsCollector             │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### 1.3 経路計画の構成方式（トークンパッシング + ECBS）

論文は Lifelong 環境で以下の構成を採用している。本仕様ではこれを**方式A**として明文化する。

```
方式A（採用方式）:
  1. 再計画が必要なエージェント群を特定する
  2. 既に確定済みの他エージェント経路は「予約制約」として
     ReservationTable に登録する
  3. 再計画対象エージェント群に対して ECBS を実行する
     - High-Level: 対象群内の衝突を制約木で解消
     - Low-Level: 各エージェント個別に Weighted A* を実行
       - ECBS からの制約 + ReservationTable の予約を同時に遵守
  4. 確定した経路を ReservationTable に追加登録する
```

トークンパッシングはタスク割当の逐次処理順を決める方式であり、ECBS は経路計画の衝突解消を行うアルゴリズムである。両者は階層が異なり、共存する。

### 1.4 メインシミュレーションループ

論文 Fig.4.1 に基づく離散時間駆動のループ。

```
初期化:
  環境・エージェント初期化
  全エージェントに初期ゴール割当
  t ← 0

ループ (t < T_max):
  Step 1: 能動的エネルギー監視
    各エージェントの危険度を判定
    危険度 ≥ 3 → 緊急介入処理

  Step 2: 充電スケジューリング
    タスク完了イベント or バッテリー警告イベント発生時:
      充電必要性を判定
      必要ならCS選択・充電タスク割当

  Step 3: ゴール割当
    ゴールを持たないエージェントに新規ゴール割当

  Step 4: 経路計画（方式A）
    再計画が必要なエージェント群を特定
    既存確定経路を ReservationTable に登録
    対象群に対し ECBS 経路計画を実行

  Step 5: エージェント移動
    各エージェントの計画済み経路に従い1 tick進行
    バッテリー残量を更新（実バッテリーで計算）

  Step 6: 状態更新・統計記録
    タスク完了判定、統計値記録
    t ← t + 1

終了処理:
  統計結果の集計・出力
```

---

## 2. 環境モデル (GridMap)

### 2.1 データ構造

```python
class GridMap:
    width: int               # マップ幅（セル数）
    height: int              # マップ高さ（セル数）
    grid: np.ndarray         # 2D配列 (height x width)
                             #   0: 通行可能
                             #   1: 障害物
    cs_positions: list[tuple[int, int]]  # CSの(y, x)座標リスト
    cs_capacity: dict[tuple[int, int], int]  # 各CSの収容台数（デフォルト1）
```

### 2.2 セル座標系

- 原点: 左上 (0, 0)
- y軸: 下方向が正
- x軸: 右方向が正
- 座標表記: (y, x) — 配列インデックスと一致させる

### 2.3 隣接関係

4近傍（上下左右）。隣接セル集合:

```
neighbors(y, x) = {(y-1,x), (y+1,x), (y,x-1), (y,x+1)}
                  ∩ {マップ範囲内} ∩ {障害物でない}
```

### 2.4 Scenario1 マップ定義（Fig.5.1からの推定座標）

論文 Fig.5.1 を読み取った30×30セルのマップ。座標は (y, x) で y=0が最上行、x=0が最左列。障害物・CS座標は画像からの推定値であり、著者実装コードが入手できた場合はそちらの値に置き換えること。

```python
SCENARIO1_WIDTH = 30
SCENARIO1_HEIGHT = 30

# 障害物: L字型（中央やや左に配置）
# Fig.5.1 から読み取り: 横長矩形(上部) + 縦長矩形(左部) が結合したL字
SCENARIO1_OBSTACLES = []
# 上部横長ブロック: y=14..16, x=8..14 (3行×7列)
for y in range(14, 17):
    for x in range(8, 15):
        SCENARIO1_OBSTACLES.append((y, x))
# 下部縦長ブロック: y=17..21, x=8..11 (5行×4列)
for y in range(17, 22):
    for x in range(8, 12):
        SCENARIO1_OBSTACLES.append((y, x))

# 充電ステーション: 3箇所（分散配置）
# Fig.5.1 の C マーク位置を読み取り
SCENARIO1_CS_POSITIONS = [
    (2, 15),   # 上部中央
    (14, 22),  # 中央右
    (28, 14),  # 下部中央
]
# 全CSの収容台数 = 1
SCENARIO1_CS_CAPACITY = {pos: 1 for pos in SCENARIO1_CS_POSITIONS}

# エージェント初期配置（互いに衝突しない位置）
# 式(4.3)の衝突条件を満たさない（チェビシェフ距離>2 or マンハッタン距離>3）
SCENARIO1_INITIAL_POSITIONS_2_AGENTS = [
    (4, 4),    # Agent 0
    (4, 25),   # Agent 1
]
SCENARIO1_INITIAL_POSITIONS_3_AGENTS = [
    (4, 4),    # Agent 0
    (4, 25),   # Agent 1
    (25, 4),   # Agent 2
]

# 全エージェントの初期回転レイヤ
INITIAL_ROTATION_LAYER = 0  # z=0（水平移動モード）

# 全エージェントの初期バッテリー
INITIAL_BATTERY = 2000.0  # B_MAX（満充電）
```

### 2.5 BFS事前計算

```python
def bfs_distance_map(grid: np.ndarray, goal: tuple[int, int]) -> np.ndarray:
    """
    ゴールから全セルへの最短距離（ステップ数）を返す。
    到達不能セルは INF。4近傍BFS。
    MoMoの3×3占有は考慮しない（セル単位の距離）。
    """

def bfs_nearest_cs(grid: np.ndarray, cs_positions: list) -> np.ndarray:
    """
    各セルから最も近いCSまでの最短距離を返す。
    複数CSの全てをBFS開始点として同時にBFSを実行（multi-source BFS）。
    """
```

**ヒューリスティックの過小評価に関する注記:**
BFS距離は3×3占有を考慮しないため、Low-Levelの実移動（`is_valid_position()`で占有チェック）より短い距離を返す場合がある。A*のヒューリスティックとしてはadmissibleなので探索の正しさには影響しないが、CS選択の `T_travel` 近似に使う場合は実際の移動時間を過小評価する可能性があることに注意。

---

## 3. エージェントモデル (Agent)

### 3.1 データ構造

```python
class Agent:
    agent_id: int
    # 現在状態
    position: tuple[int, int]     # (y, x) 中心位置
    rotation_layer: int           # z ∈ {0, 1, 2, 3, 4, 5}
    battery: float                # b ∈ [0, B_MAX] (実数値、離散化しない)
    # タスク管理
    user_task_queue: deque         # W_i: ユーザタスク列
    execution_task_list: list      # T_i: 実行タスク列（充電タスク含む）
    current_goal: tuple[int, int] | None  # 現在の目標位置
    current_goal_type: str        # "task" or "charging"
    # 経路
    planned_timeline: dict[int, TimedOccupancy]  # tick → 占有情報（後述）
    planned_actions: list[ActionEntry]  # イベント列
    path_index: int               # 現在の実行位置
    # 統計
    completed_tasks: int          # 完了ユーザタスク数（充電タスクを含めない）
    total_charge_count: int       # 充電回数
    # バッテリー履歴（緊急介入の減少率予測用）
    battery_history: list[tuple[int, float]]  # [(tick, battery), ...]
```

### 3.2 ゴール完了条件の厳密定義

```python
def is_goal_reached(agent: Agent) -> bool:
    """
    ゴール到達判定。ゴールタイプにより条件が異なる。

    task ゴール:
      position == goal_pos かつ battery >= min_goal_battery
      完了タスク数 completed_tasks += 1

    charging ゴール:
      position == cs_pos かつ battery >= B_MAX * CHARGE_TARGET_THRESHOLD (80%)
      完了タスク数にはカウントしない
    """
    if agent.current_goal_type == "task":
        return (agent.position == agent.current_goal and
                agent.battery >= agent.min_goal_battery)
    elif agent.current_goal_type == "charging":
        return (agent.position == agent.current_goal and
                agent.battery >= B_MAX * CHARGE_TARGET_THRESHOLD)
```

### 3.3 MoMoサイズと占有領域

MoMo5は中心位置を基準に約3×3セルを占有する。衝突判定は式(4.3)の距離ベース判定を使用する。

### 3.4 回転レイヤの定義

MoMo5の方向転換は6段階（L=6）の回転レイヤで表現する。

```
z = 0: 水平移動用レイヤ（左右移動が可能）
z = 1: 中間レイヤ1（移動不可、回転のみ）
z = 2: 中間レイヤ2（移動不可、回転のみ）
z = 3: 中間レイヤ3（移動不可、回転のみ）
z = 4: 中間レイヤ4（移動不可、回転のみ）
z = 5: 垂直移動用レイヤ（上下移動が可能）
```

レイヤに応じた移動可能方向:

```python
def get_movable_directions(z: int) -> list[tuple[int, int]]:
    if z == 0:
        return [(0, -1), (0, 1)]   # 左, 右
    elif z == 5:
        return [(-1, 0), (1, 0)]   # 上, 下
    else:
        return []                   # 中間レイヤでは移動不可
```

回転の遷移:
- z → z+1 または z → z-1（境界チェックあり）
- z=0からはz=1へのみ、z=5からはz=4へのみ回転可能

---

## 4. バッテリーモデル (BatteryModel)

### 4.1 パラメータ定数

```python
# 論文 Table 5.2 より
B_MAX = 2000           # バッテリー容量 [単位]
DELTA_B_MOVE = 5.0     # 移動1ステップあたりの消費
DELTA_B_ROT_TOTAL = 0.5  # 方向転換全体（z=0↔z=5）のバッテリー消費
DELTA_B_ROT_STEP = 0.1   # 1レイヤ遷移（z→z±1）あたりの消費 = 0.5/5
DELTA_B_WAIT = 0.0     # 待機時の消費（ゼロ）

C_MOVE = 1             # 移動の時間コスト [tick] （整数）
C_WAIT = 1             # 待機の時間コスト [tick] （整数）

# 回転コスト:
# 論文では方向転換全体（水平→垂直）のコストを c(Rotate) = c_rot ≈ 5c_move と定義。
# L=6レイヤ表現では z=0→z=5 に5段階の遷移が必要。
# したがって、1レイヤ遷移あたりのコストは以下の通り:
C_ROT_TOTAL = 5        # 方向転換全体の時間コスト [tick]（z=0↔z=5）
DELTA_B_ROT_TOTAL = 0.5  # 方向転換全体のバッテリー消費

C_ROT_STEP = 1         # 1レイヤ遷移(z→z±1)の時間コスト [tick] = C_ROT_TOTAL / 5
DELTA_B_ROT_STEP = 0.1  # 1レイヤ遷移のバッテリー消費 = DELTA_B_ROT_TOTAL / 5

# 区分線形充電レート
R_FAST = 8.0           # SoC 0–40% 区間 [単位/tick]
R_MEDIUM = 4.0         # SoC 40–80% 区間 [単位/tick]
R_SLOW = 2.0           # SoC 80–100% 区間 [単位/tick]

# SoC閾値
SOC_FAST_THRESHOLD = 0.40    # 40%
SOC_MEDIUM_THRESHOLD = 0.80  # 80%

# 充電スケジューリング閾値
CHARGE_START_THRESHOLD = 0.30   # 30%（この閾値以下で充電開始）
CHARGE_TARGET_THRESHOLD = 0.80  # 80%（この閾値まで充電）

# ゴール到着時最低残量（デフォルト値。PlanningRequest で上書き可能）
B_GOAL_DEFAULT = 10.0
```

### 4.2 充電レート関数

```python
def charge_rate(current_battery: float) -> float:
    soc = current_battery / B_MAX
    if soc < SOC_FAST_THRESHOLD:
        return R_FAST
    elif soc < SOC_MEDIUM_THRESHOLD:
        return R_MEDIUM
    else:
        return R_SLOW
```

### 4.3 充電時間計算

```python
def compute_charge_time(current_b: float, target_b: float) -> int:
    """
    current_b から target_b まで充電するのに必要なtick数を返す。
    区間をまたぐ場合は各区間のレートを個別に適用。
    戻り値は整数（切り上げ）。
    """
    boundary_40 = B_MAX * SOC_FAST_THRESHOLD    # 800
    boundary_80 = B_MAX * SOC_MEDIUM_THRESHOLD  # 1600

    total_time = 0.0
    b = current_b

    if b < boundary_40 and target_b > b:
        upper = min(target_b, boundary_40)
        total_time += (upper - b) / R_FAST
        b = upper

    if b < boundary_80 and target_b > b:
        upper = min(target_b, boundary_80)
        total_time += (upper - b) / R_MEDIUM
        b = upper

    if b < B_MAX and target_b > b:
        upper = min(target_b, B_MAX)
        total_time += (upper - b) / R_SLOW
        b = upper

    return math.ceil(total_time)
```

### 4.4 充電後バッテリー残量計算

```python
def compute_battery_after_charge(current_b: float, charge_ticks: int) -> float:
    """
    current_b からcharge_ticks tick分充電した後のバッテリー残量を返す。
    区分線形充電の各区間境界をまたいで正しく積分する。

    アルゴリズム:
    1. 残り充電時間 remaining = charge_ticks
    2. 現在区間の充電レートを取得
    3. 現在区間の上限までの充電時間を計算
    4. min(remaining, 区間内時間) だけ充電
    5. 次の区間へ進み、remaining > 0 なら繰り返す
    """
    boundary_40 = B_MAX * SOC_FAST_THRESHOLD
    boundary_80 = B_MAX * SOC_MEDIUM_THRESHOLD
    boundaries = [boundary_40, boundary_80, B_MAX]

    b = current_b
    remaining = float(charge_ticks)

    for upper_bound in boundaries:
        if remaining <= 0 or b >= B_MAX:
            break
        if b >= upper_bound:
            continue
        rate = charge_rate(b)
        capacity_in_segment = upper_bound - b
        time_for_segment = capacity_in_segment / rate
        actual_time = min(remaining, time_for_segment)
        b += rate * actual_time
        remaining -= actual_time

    return min(b, B_MAX)
```

**充電の切り上げオーバーシュートに関する注記:**
`compute_charge_time()` は `ceil()` で切り上げた整数tick数を返す。`compute_battery_after_charge()` はこの切り上げ後のtick数で充電するため、探索ノードの `target_level` を少し超える場合がある。これは実行時モデルとしては自然であり、安全性に問題はない（バッテリーが多い分には安全）。ただし、SoC離散化と組み合わせると、離散化レベルが1段上にずれる可能性がある。この影響は微小だが、実験2の粗粒度設定で結果に差が出る場合はこの点を確認すること。

### 4.5 SoC離散化

```python
def discretize_battery(b: float, delta_soc: float) -> float:
    """
    バッテリー残量を離散化レベルに丸める。
    式(4.46): b̃ = round(b / ΔSoC) * ΔSoC

    注意: delta_soc はPlannerインスタンスから渡される。
    グローバル定数を使わず、L_SoC設定に応じて変化する。

    丸め規則について:
    - 論文式(4.46)は round() を使用
    - 消費後の過大評価が問題になる場合は floor() に変更する
      オプションを config で提供
    - デフォルトは round()（論文準拠）
    """
    return round(b / delta_soc) * delta_soc
```

### 4.6 バッテリー遷移規則（式4.10）

```python
def battery_transition(b: float, action: str, charge_ticks: int = 0) -> float:
    """シミュレーション実行層でのバッテリー更新（実数値で計算）"""
    if action == "move":
        return b - DELTA_B_MOVE
    elif action == "rotate_step":
        return b - DELTA_B_ROT_STEP
    elif action == "wait":
        return b - DELTA_B_WAIT
    elif action == "charge":
        return compute_battery_after_charge(b, charge_ticks)
    else:
        raise ValueError(f"Unknown action: {action}")
```

---

## 5. 時間占有モデル（中間tick占有の明示化）

### 5.1 問題

Rotate は水平↔垂直の完全方向転換で合計5 tick（1レイヤ遷移あたり C_ROT_STEP=1 tick × 5段階）、Charge は可変tick を消費する。探索状態は開始tickと終了tickのみ保持するが、その間の全tickでエージェントは同一セルを占有している。他エージェントがこの中間tickですり抜けないよう、全tickの占有列を明示的に管理する必要がある。

### 5.2 データ構造

```python
@dataclass
class TimedOccupancy:
    """1 tickにおけるエージェントの占有情報"""
    y: int
    x: int
    z: int
    action: str      # この tick で実行中の行動
    battery: float   # この tick でのバッテリー残量

@dataclass
class ActionEntry:
    """1つの行動イベント"""
    action: str           # "move", "rotate", "wait", "charge"
    start_tick: int       # 行動開始tick
    end_tick: int         # 行動終了tick（この tick に次状態へ遷移）
    start_pos: tuple[int, int]
    end_pos: tuple[int, int]
    start_z: int
    end_z: int
    battery_before: float
    battery_after: float
```

### 5.3 経路から占有タイムラインへの展開

```python
def expand_to_timeline(actions: list[ActionEntry]) -> dict[int, TimedOccupancy]:
    """
    行動列を全tickの占有マップに展開する。

    規則:
    - Move (C_MOVE=1):
        start_tick → start_pos を占有
        end_tick   → end_pos を占有（次の行動の start_tick と同一）
    - Rotate (C_ROT_STEP=1 per layer):
        start_tick ~ end_tick-1 の間、同一セルを占有
        （1レイヤ遷移ごとに1tick。完全方向転換は5回のRotateイベント）
    - Wait (C_WAIT=1):
        start_tick を占有
    - Charge (可変tick):
        start_tick ~ end_tick-1 まで、同一CS位置を占有

    終端tick:
    最後の行動の end_tick は明示的に登録する（ゴール到達tick）。
    登録しないと衝突検出・ゴール判定・統計がずれる。

    バッテリー精度:
    各tickのバッテリーは行動の battery_before/after から線形補間する。
    安全性検証（b(t)≥0, b_goal≥B_min）で正確な値が必要なため、
    後処理でも行動列から正確に復元可能とする。
    """
    timeline = {}
    for entry in actions:
        duration = entry.end_tick - entry.start_tick

        if entry.action == "move":
            # Move: start_tick に移動元を登録
            timeline[entry.start_tick] = TimedOccupancy(
                y=entry.start_pos[0], x=entry.start_pos[1],
                z=entry.start_z, action="move",
                battery=entry.battery_before,
            )
            # end_tick は次の行動の start_tick と同一なので、
            # 次の行動で上書きされる。最後の行動の場合は後述の終端処理で登録。

        else:
            # Rotate / Wait / Charge: 全中間tick で同一セルを占有
            for t in range(entry.start_tick, entry.end_tick):
                progress = (t - entry.start_tick) / max(duration, 1)
                interp_battery = (entry.battery_before +
                                  (entry.battery_after - entry.battery_before) * progress)
                timeline[t] = TimedOccupancy(
                    y=entry.start_pos[0], x=entry.start_pos[1],
                    z=entry.start_z, action=entry.action,
                    battery=interp_battery,
                )

    # 終端tick: 最後の行動の end_tick を必ず登録
    if actions:
        last = actions[-1]
        timeline[last.end_tick] = TimedOccupancy(
            y=last.end_pos[0], x=last.end_pos[1],
            z=last.end_z, action="arrive",
            battery=last.battery_after,
        )

    return timeline
```

### 5.4 PlanResult の改訂

```python
@dataclass
class PlanResult:
    actions: list[ActionEntry]              # イベント列
    timeline: dict[int, TimedOccupancy]     # 全tick占有マップ（展開済み）
    cost: float                             # 総コスト（= 最終tick - 開始tick）
    stats: PlanStats

    @property
    def path(self) -> list[tuple[int, int]]:
        """各tickの位置列を返す（後方互換用）"""
        return [(self.timeline[t].y, self.timeline[t].x)
                for t in sorted(self.timeline.keys())]
```

---

## 6. 衝突判定 (ConflictDetector)

### 6.1 頂点衝突（式4.3）

```python
def is_vertex_conflict(
    pos_i: tuple[int, int],
    pos_j: tuple[int, int]
) -> bool:
    """
    条件: max(Δx, Δy) ≤ 2 かつ Δx + Δy ≤ 3
    """
    dy = abs(pos_i[0] - pos_j[0])
    dx = abs(pos_i[1] - pos_j[1])
    return max(dx, dy) <= 2 and (dx + dy) <= 3
```

### 6.2 辺衝突

```python
def is_edge_conflict(
    pos_i_t: tuple[int, int],
    pos_i_t1: tuple[int, int],
    pos_j_t: tuple[int, int],
    pos_j_t1: tuple[int, int]
) -> bool:
    return (is_vertex_conflict(pos_i_t, pos_j_t1) or
            is_vertex_conflict(pos_j_t, pos_i_t1))
```

### 6.3 衝突データ構造

```python
@dataclass
class Conflict:
    type: str               # "vertex", "edge", "cs_capacity"
    agent_i: int
    agent_j: int
    timestep: int
    # 頂点衝突: 両エージェントの位置を保持
    # MoMoの衝突は距離条件であり、2台は異なるセルにいる場合もある
    agent_i_pos: tuple[int, int] | None = None
    agent_j_pos: tuple[int, int] | None = None
    # 辺衝突
    agent_i_from: tuple[int, int] | None = None
    agent_i_to: tuple[int, int] | None = None
    agent_j_from: tuple[int, int] | None = None
    agent_j_to: tuple[int, int] | None = None
    # CS容量衝突
    cs_position: tuple[int, int] | None = None
```

### 6.4 制約データ構造（修正版）

```python
@dataclass
class Constraint:
    agent_id: int
    type: str  # "vertex", "edge", "range", "cs_capacity"
    timestep: int
    # 頂点制約: position が必須
    position: tuple[int, int] | None = None
    # 辺制約: from_pos → to_pos の遷移を禁止
    from_pos: tuple[int, int] | None = None
    to_pos: tuple[int, int] | None = None
    # RANGE制約: time_start ≤ t ≤ time_end で position を禁止
    time_start: int | None = None
    time_end: int | None = None
```

### 6.5 全衝突検出（タイムラインベース）

```python
def find_first_conflict(
    timelines: dict[int, dict[int, TimedOccupancy]],  # agent_id → timeline
    cs_positions: list,
    cs_capacity: dict
) -> Conflict | None:
    """
    全エージェントペアについて、タイムラインを用いて
    最も早い時刻の衝突を1つ返す。

    タイムラインベースなので、Rotate/Charge中の中間tickも検査される。
    """
    agent_ids = list(timelines.keys())
    all_ticks = set()
    for tl in timelines.values():
        all_ticks.update(tl.keys())

    for t in sorted(all_ticks):
        # 頂点衝突
        for i_idx in range(len(agent_ids)):
            for j_idx in range(i_idx + 1, len(agent_ids)):
                ai, aj = agent_ids[i_idx], agent_ids[j_idx]
                if t not in timelines[ai] or t not in timelines[aj]:
                    continue
                oi, oj = timelines[ai][t], timelines[aj][t]
                if is_vertex_conflict((oi.y, oi.x), (oj.y, oj.x)):
                    return Conflict("vertex", ai, aj, t,
                                    agent_i_pos=(oi.y, oi.x),
                                    agent_j_pos=(oj.y, oj.x))

        # 辺衝突（tick t → t+1）
        t1 = t + 1
        for i_idx in range(len(agent_ids)):
            for j_idx in range(i_idx + 1, len(agent_ids)):
                ai, aj = agent_ids[i_idx], agent_ids[j_idx]
                if (t not in timelines[ai] or t1 not in timelines[ai] or
                    t not in timelines[aj] or t1 not in timelines[aj]):
                    continue
                pi_t = (timelines[ai][t].y, timelines[ai][t].x)
                pi_t1 = (timelines[ai][t1].y, timelines[ai][t1].x)
                pj_t = (timelines[aj][t].y, timelines[aj][t].x)
                pj_t1 = (timelines[aj][t1].y, timelines[aj][t1].x)
                if is_edge_conflict(pi_t, pi_t1, pj_t, pj_t1):
                    return Conflict("edge", ai, aj, t,
                                    agent_i_from=pi_t, agent_i_to=pi_t1,
                                    agent_j_from=pj_t, agent_j_to=pj_t1)

    # CS容量衝突
    for t in sorted(all_ticks):
        for cs_pos in cs_positions:
            agents_at_cs = [
                aid for aid in agent_ids
                if t in timelines[aid] and
                   (timelines[aid][t].y, timelines[aid][t].x) == cs_pos
            ]
            if len(agents_at_cs) > cs_capacity.get(cs_pos, 1):
                return Conflict("cs_capacity", agents_at_cs[0],
                                agents_at_cs[1], t, cs_position=cs_pos)
    return None
```

### 6.6 CS接近渋滞の検出とRange Constraint

**デフォルト: OFF（論文再現モードでは使用しない）**

論文では、CS容量超過は高レベル制約として扱う一方、CS接近渋滞は高レベル衝突としては扱わず、実行層の待ち行列推定や退避処理で対処すると説明されている。したがって論文再現モード（モード1/モード2）ではこの機構を無効化する。

```python
# config.py
ENABLE_CS_CONGESTION_RANGE_CONSTRAINT = False  # デフォルト OFF

# CS接近渋滞の検出パラメータ（拡張実験モード用）
CS_CONGESTION_RADIUS = 3
CS_CONGESTION_CONFLICT_THRESHOLD = 3
```

拡張実験として有効化する場合のみ、以下のロジックを使用する:

```python
def detect_cs_congestion(
    conflict_history: list[Conflict],
    cs_positions: list[tuple[int, int]]
) -> list[Constraint]:
    """
    拡張実験モードでのみ使用。
    CS周辺で繰り返し衝突が発生した場合に Range Constraint を生成。
    """
    if not ENABLE_CS_CONGESTION_RANGE_CONSTRAINT:
        return []
    # ...
```

---

## 7. ヒューリスティック関数 (Heuristic)

### 7.1 クラス設計

```python
class EnergyAwareHeuristic:
    def __init__(
        self,
        grid_map: GridMap,
        goal: tuple[int, int],
        min_goal_battery: float,  # B_goal（リクエストごとに異なりうる）
        delta_soc: float          # L_SoC に依存する離散化幅
    ):
        self.goal = goal
        self.min_goal_battery = min_goal_battery
        self.delta_soc = delta_soc
        self.d_goal = bfs_distance_map(grid_map.grid, goal)
        self.d_cs = bfs_nearest_cs(grid_map.grid, grid_map.cs_positions)
        self._cache = {}
```

### 7.2 ヒューリスティック値計算（式4.33–4.44）

```python
def compute(self, y: int, x: int, battery: float) -> float:
    level = int(round(battery / self.delta_soc))
    cache_key = (y, x, level, self.goal[0], self.goal[1])
    if cache_key in self._cache:
        return self._cache[cache_key]

    d_goal = self.d_goal[y, x]
    d_cs = self.d_cs[y, x]

    if d_goal == INF:
        self._cache[cache_key] = INF
        return INF

    t_goal = d_goal * C_MOVE
    e_goal = d_goal * DELTA_B_MOVE
    t_cs = d_cs * C_MOVE
    e_cs = d_cs * DELTA_B_MOVE

    # 必要エネルギー（式4.39）— B_GOAL をリクエストから取得
    e_need = e_goal + self.min_goal_battery

    # 直接到達ヒューリスティック（式4.40）
    h_direct = t_goal if battery >= e_need else INF

    # CS経由ヒューリスティック（式4.41–4.43）
    if d_cs < INF:
        b_at_cs = battery - e_cs
        if b_at_cs < 0:
            h_cs_val = INF
        else:
            r_avg = 0.4 * R_FAST + 0.4 * R_MEDIUM + 0.2 * R_SLOW
            if b_at_cs >= e_need:
                t_charge = 0.0
            else:
                t_charge = (e_need - b_at_cs) / r_avg
            h_cs_val = t_cs + t_charge + t_goal
    else:
        h_cs_val = INF

    h_val = min(h_direct, h_cs_val)
    self._cache[cache_key] = h_val
    return h_val
```

---

## 8. Low-Level Search（Weighted A*）

### 8.1 計画リクエスト

```python
@dataclass
class PlanningRequest:
    """Low-Level探索への入力を一元化"""
    agent_id: int
    start_pos: tuple[int, int]
    start_z: int
    start_battery: float          # battery_raw（実数値）
    start_time: int
    goal_pos: tuple[int, int]
    goal_type: str                 # "task" or "charging"
    min_goal_battery: float        # ゴール到着時最低残量
    constraints: list[Constraint]
    reservation_table: ReservationTable
```

**goal_type ごとの min_goal_battery 設定:**
```python
# task ゴールの場合:
request = PlanningRequest(
    ...,
    goal_type="task",
    min_goal_battery=B_GOAL_DEFAULT,  # 10.0（ケースごとに変更可）
)

# charging ゴールの場合:
request = PlanningRequest(
    ...,
    goal_type="charging",
    min_goal_battery=B_MAX * CHARGE_TARGET_THRESHOLD,  # 1600.0 (80%)
)
```

**Low-Levelゴール判定（Algorithm 1 行14に対応）:**
```python
# メインループ内でのゴール判定
if (state.y, state.x) == request.goal_pos:
    if node.battery_raw >= request.min_goal_battery:
        # 解発見 → 経路を復元して返す
        return reconstruct_path(node)
```

`battery_raw`（実数値）で判定する。`state.battery`（離散化値）では判定しない。

### 8.2 探索状態

```python
@dataclass(frozen=True)
class SearchState:
    y: int                 # 中心位置 y座標
    x: int                 # 中心位置 x座標
    z: int                 # 回転レイヤ {0, 1, 2, 3, 4, 5}
    battery: float         # バッテリー残量（離散化済み）
    timestep: int          # 離散時刻

    def state_key(self, delta_soc: float) -> tuple:
        """重複排除用キー（式4.48）— delta_socを引数で受ける"""
        return (self.y, self.x, self.z, self.timestep,
                int(self.battery // delta_soc))
```

### 8.3 探索ノード

```python
@dataclass
class SearchNode:
    state: SearchState
    g: float               # 開始からのコスト
    h: float               # ヒューリスティック値
    f: float               # f = g + w_LL * h
    parent: SearchNode | None
    action: str            # この状態に至った行動
    battery_raw: float     # 離散化前のバッテリー残量（実数値）
```

### 8.4 LowLevelPlanner

```python
class LowLevelPlanner:
    def __init__(
        self,
        grid_map: GridMap,
        w_ll: float = 1.0,
        max_expansions: int = 5000,
        max_generations: int = 10000,
        l_soc: int = 100,
        disable_occupancy_check: bool = False  # 実験1・2で True
    ):
        self.grid_map = grid_map
        self.w_ll = w_ll
        self.max_expansions = max_expansions
        self.max_generations = max_generations
        self.l_soc = l_soc
        self.delta_soc = B_MAX / l_soc
        self.disable_occupancy_check = disable_occupancy_check

    def plan(self, request: PlanningRequest) -> PlanResult | None:
        """Algorithm 1 の実装。"""
```

`disable_occupancy_check=True` の場合、`is_valid_position()` によるMoMo 3×3占有チェックをバイパスし、単純なセル単位の障害物チェック（`grid[ny][nx] != 1`）のみを行う。実験1・2（シングルエージェントの安全性フィルタ単体検証）で使用する。実験3（マルチエージェント環境）では `False`（デフォルト）。

### 8.5 遷移関数

**バッテリー更新の原則:**
```
- 安全性判定（b(t)≥0, b_goal≥B_min, S1/S2）は battery_raw（実数値）を用いる。
- SearchState.battery は離散化済み値であり、探索キー・h計算・ドミナンス判定にのみ使う。
- 次状態の実バッテリーは必ず親ノードの SearchNode.battery_raw から計算する。
- 離散化値から消費すると丸め誤差が蓄積するため、必ず raw → raw → ... で伝播する。
```

**Move遷移（式4.17–4.20）:**
```python
def transition_move(node: SearchNode, direction: tuple[int,int], delta_soc: float):
    s = node.state
    ny, nx = s.y + direction[0], s.x + direction[1]
    raw_b = node.battery_raw - DELTA_B_MOVE
    new_t = s.timestep + C_MOVE  # +1
    return SearchState(ny, nx, s.z,
                       discretize_battery(raw_b, delta_soc), new_t), raw_b
```

**Rotate遷移（式4.21–4.24 — 1レイヤ段階遷移）:**
```python
def transition_rotate(node: SearchNode, dz: int, delta_soc: float):
    """
    z → z±1 の1段階遷移。
    時間コスト: C_ROT_STEP = 1 tick
    バッテリー消費: DELTA_B_ROT_STEP = 0.1
    水平→垂直(z=0→z=5)の完全方向転換には5回呼ばれ、
    合計 5 tick / 0.5 battery となる。
    """
    s = node.state
    new_z = s.z + dz  # +1 or -1, 境界チェック要
    raw_b = node.battery_raw - DELTA_B_ROT_STEP
    new_t = s.timestep + C_ROT_STEP  # +1
    return SearchState(s.y, s.x, new_z,
                       discretize_battery(raw_b, delta_soc), new_t), raw_b
```

**Wait遷移（式4.25–4.28）:**
```python
def transition_wait(node: SearchNode, delta_soc: float):
    s = node.state
    raw_b = node.battery_raw - DELTA_B_WAIT
    new_t = s.timestep + C_WAIT  # +1
    return SearchState(s.y, s.x, s.z,
                       discretize_battery(raw_b, delta_soc), new_t), raw_b
```

**Charge遷移（式4.29–4.32）:**
```python
def transition_charge(node: SearchNode, target_level: float, delta_soc: float):
    s = node.state
    charge_ticks = compute_charge_time(node.battery_raw, target_level)
    raw_b = compute_battery_after_charge(node.battery_raw, charge_ticks)
    new_t = s.timestep + charge_ticks
    return SearchState(s.y, s.x, s.z,
                       discretize_battery(raw_b, delta_soc), new_t), raw_b
```

### 8.6 gコストの計算

```python
# 各行動のgコスト加算:
# Move:   g += C_MOVE        (= 1)
# Rotate: g += C_ROT_STEP    (= 1, 1レイヤ段階遷移)
# Wait:   g += C_WAIT        (= 1)
# Charge: g += charge_ticks  (= compute_charge_time の戻り値、整数)
#
# 注: z=0→z=5 の完全方向転換は5回のRotateで g += 5
```

### 8.7 制約違反チェック（修正版：中間tick対応）

Low-Level探索では、遷移の**終端tick**だけでなく、Rotate/Charge/Waitの**全中間tick**で制約違反をチェックする必要がある。終端tickのみチェックすると、例えばt=10→15のRotate中にt=12の制約を見逃す。

```python
def violates_constraints_during_action(
    start_pos: tuple[int, int],
    end_pos: tuple[int, int],
    start_tick: int,
    end_tick: int,
    action: str,
    constraints: list[Constraint],
    agent_id: int
) -> bool:
    """
    行動の全tick（start_tick ～ end_tick）で制約違反がないかチェック。
    Move以外の行動では全中間tickで同一セルに留まるため、
    その全tickについて vertex / range 制約を検査する。
    Move の場合は start_tick に from_pos、end_tick に to_pos。
    """
    for t in range(start_tick, end_tick + 1):
        # このtickでの位置を決定
        if action == "move":
            pos = start_pos if t == start_tick else end_pos
        else:
            # rotate, wait, charge は全tick同一セル
            pos = start_pos

        for c in constraints:
            if c.agent_id != agent_id:
                continue

            # 頂点制約
            if c.type == "vertex":
                if c.timestep == t and c.position == pos:
                    return True

            # 辺制約（Move遷移時のみ、遷移tickで判定）
            elif c.type == "edge":
                if (action == "move" and t == start_tick and
                    c.timestep == t and
                    c.from_pos == start_pos and c.to_pos == end_pos):
                    return True

            # RANGE制約
            elif c.type == "range":
                if (c.position == pos and
                    c.time_start <= t <= c.time_end):
                    return True

    return False
```

**Low-Level探索のメインループでの使用:**
```
各行動 α の遷移先 s' を生成した後:
  if violates_constraints_during_action(
      start_pos=(s.y, s.x), end_pos=(s'.y, s'.x),
      start_tick=s.timestep, end_tick=s'.timestep,
      action=α, constraints=constraints, agent_id=agent_id):
      continue  # この遷移を棄却
```

### 8.8 予約テーブルとの衝突チェック（中間tick + 辺衝突対応）

```python
def violates_reservations_during_action(
    start_pos: tuple[int, int],
    end_pos: tuple[int, int],
    start_tick: int,
    end_tick: int,
    action: str,
    agent_id: int,
    reservation_table: ReservationTable
) -> bool:
    """
    行動の全中間tickで予約テーブルとの頂点衝突がないかチェック。
    さらにMove遷移では辺衝突もチェック。
    """
    # 頂点衝突チェック（全中間tick）
    for t in range(start_tick, end_tick + 1):
        if action == "move":
            pos = start_pos if t == start_tick else end_pos
        else:
            pos = start_pos
        if reservation_table.is_vertex_reserved(pos, t, agent_id):
            return True

    # 辺衝突チェック（Move遷移時のみ）
    if action == "move":
        if reservation_table.is_edge_reserved(
            start_pos, end_pos, start_tick, agent_id):
            return True

    return False
```

### 8.9 戦略的バッテリレベル

```python
def compute_strategic_battery_levels(
    current_b: float,
    goal_pos: tuple[int, int],
    current_pos: tuple[int, int],
    heuristic: EnergyAwareHeuristic,
    delta_soc: float
) -> list[float]:
    levels = []
    d_goal = heuristic.d_goal[current_pos[0], current_pos[1]]
    e_need = d_goal * DELTA_B_MOVE + heuristic.min_goal_battery

    level_min = e_need
    if level_min > current_b and level_min <= B_MAX:
        levels.append(min(level_min, B_MAX))

    level_proactive = B_MAX * CHARGE_TARGET_THRESHOLD  # 1600
    if level_proactive > current_b and level_proactive not in levels:
        levels.append(level_proactive)

    level_full = B_MAX
    if level_full > current_b and level_full not in levels:
        levels.append(level_full)

    return [discretize_battery(l, delta_soc) for l in levels]
```

### 8.10 3×3占有チェック

```python
def is_valid_position(grid: np.ndarray, center_y: int, center_x: int) -> bool:
    h, w = grid.shape
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            ny, nx = center_y + dy, center_x + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                return False
            if grid[ny, nx] == 1:
                return False
    return True
```

### 8.11 探索結果の統計

```python
@dataclass
class PlanStats:
    expanded: int          # 展開ノード数
    generated: int         # 生成ノード数
    time_ms: float         # 探索時間 [ms]
    charge_actions: int    # 充電アクション数
    charge_nodes_generated: int  # 探索中に生成された充電ノード数
    rotate_actions: int    # 回転回数
    path_length: int       # 経路長（ステップ数）
    terminated_by: str     # "solution", "no_solution", "cutoff"
```

---

## 9. High-Level Search（ECBS）

### 9.1 制約木ノード

```python
@dataclass
class CTNode:
    constraints: list[Constraint]
    paths: dict[int, PlanResult]
    cost: float                           # f(N) = Σ cost(π_i)
    conflicts: int                        # 衝突数（FOCAL用二次基準）
    parent: CTNode | None
```

### 9.2 ECBSアルゴリズム

```python
class ECBSPlanner:
    def __init__(
        self,
        grid_map: GridMap,
        w_hl: float = 1.5,
        w_ll: float = 1.5,
        l_soc: int = 100,
    ):
        self.grid_map = grid_map
        self.w_hl = w_hl
        self.low_level = LowLevelPlanner(grid_map, w_ll, l_soc=l_soc)

    def plan(
        self,
        requests: list[PlanningRequest],      # 再計画対象エージェント群
        existing_reservations: ReservationTable  # 既存確定経路の予約
    ) -> dict[int, PlanResult] | None:
        """
        方式A: 対象エージェント群を ECBS で同時に解く。
        既存エージェントの確定経路は reservation_table 経由で
        Low-Level に渡す。
        """
```

### 9.3 FOCAL Search

```python
class FocalList:
    def __init__(self, w_hl: float):
        self.w_hl = w_hl
        self.open_heap = []
        self.focal_heap = []

    def update_focal(self):
        f_min = self.open_heap[0][0] if self.open_heap else INF
        threshold = self.w_hl * f_min
        self.focal_heap = [
            (n.conflicts, tb, n)
            for f, tb, n in self.open_heap
            if f <= threshold
        ]
        heapq.heapify(self.focal_heap)
```

### 9.4 衝突から制約への変換

ECBS の衝突解消では、検出された Conflict から2つの子ノード用の Constraint を生成する。MoMo の距離ベース衝突条件では、2台のエージェントは**異なるセル**にいても衝突が発生するため、各エージェントに**自分の位置**を禁止する制約を課す。

```python
def generate_constraints(conflict: Conflict) -> list[tuple[int, Constraint]]:
    """
    衝突から制約を生成する。
    各タプルは (分岐子ノード番号, Constraint)。
    子1: agent_i に制約を課す
    子2: agent_j に制約を課す

    Returns:
        [(0, constraint_for_i), (1, constraint_for_j)]
    """
    results = []

    if conflict.type == "vertex":
        # 子1: agent_i の位置 agent_i_pos を時刻 t に禁止
        results.append((0, Constraint(
            agent_id=conflict.agent_i,
            type="vertex",
            timestep=conflict.timestep,
            position=conflict.agent_i_pos,
        )))
        # 子2: agent_j の位置 agent_j_pos を時刻 t に禁止
        results.append((1, Constraint(
            agent_id=conflict.agent_j,
            type="vertex",
            timestep=conflict.timestep,
            position=conflict.agent_j_pos,
        )))

    elif conflict.type == "edge":
        # 子1: agent_i の遷移 from→to を時刻 t に禁止
        results.append((0, Constraint(
            agent_id=conflict.agent_i,
            type="edge",
            timestep=conflict.timestep,
            from_pos=conflict.agent_i_from,
            to_pos=conflict.agent_i_to,
        )))
        # 子2: agent_j の遷移 from→to を時刻 t に禁止
        results.append((1, Constraint(
            agent_id=conflict.agent_j,
            type="edge",
            timestep=conflict.timestep,
            from_pos=conflict.agent_j_from,
            to_pos=conflict.agent_j_to,
        )))

    elif conflict.type == "cs_capacity":
        # CS容量衝突: capacity=1 の場合、
        # 同時にCS上にいる2台のうち片方ずつに
        # cs_position at t を禁止する子ノードを生成
        results.append((0, Constraint(
            agent_id=conflict.agent_i,
            type="vertex",
            timestep=conflict.timestep,
            position=conflict.cs_position,
        )))
        results.append((1, Constraint(
            agent_id=conflict.agent_j,
            type="vertex",
            timestep=conflict.timestep,
            position=conflict.cs_position,
        )))

    return results
```

**ECBS子ノード生成の手順:**
```
1. conflict = find_first_conflict(N.paths)
2. constraint_pairs = generate_constraints(conflict)
3. for (branch_id, constraint) in constraint_pairs:
     child = copy(N)
     child.constraints.append(constraint)
     agent = constraint.agent_id
     child.paths[agent] = low_level.plan(agent, child.constraints)
     if child.paths[agent] is not None:
         child.cost = sum(cost(π) for π in child.paths.values())
         child.conflicts = count_conflicts(child.paths)
         OPEN.add(child)
     else:
         child.cost = ∞  # 暗黙的枝刈り
```

---

## 10. 予約テーブル

### 10.1 データ構造

```python
class ReservationTable:
    def __init__(self):
        # 頂点予約: (timestep, y, x) → agent_id
        self.vertex_reservations: dict[tuple[int, int, int], int] = {}
        # 辺予約: (timestep, y1, x1, y2, x2) → agent_id
        #   「tick t に (y1,x1) から (y2,x2) へ移動中」を表す
        self.edge_reservations: dict[tuple[int, int, int, int, int], int] = {}

    def add_timeline(
        self,
        agent_id: int,
        timeline: dict[int, TimedOccupancy]
    ):
        """タイムラインの全tickを頂点予約に登録し、Moveの辺も登録"""
        sorted_ticks = sorted(timeline.keys())
        for t in sorted_ticks:
            occ = timeline[t]
            self.vertex_reservations[(t, occ.y, occ.x)] = agent_id

        # 連続tickで位置が変化している場合は辺予約を登録
        for i in range(len(sorted_ticks) - 1):
            t = sorted_ticks[i]
            t1 = sorted_ticks[i + 1]
            if t1 == t + 1:  # 隣接tickの場合のみ
                occ_t = timeline[t]
                occ_t1 = timeline[t1]
                if (occ_t.y, occ_t.x) != (occ_t1.y, occ_t1.x):
                    self.edge_reservations[
                        (t, occ_t.y, occ_t.x, occ_t1.y, occ_t1.x)
                    ] = agent_id

    def is_vertex_reserved(
        self,
        pos: tuple[int, int],
        timestep: int,
        exclude_agent: int
    ) -> bool:
        """
        3×3占有を考慮した頂点予約チェック。
        式(4.3)の条件で既存予約と衝突するか判定。
        """
        for (t, ry, rx), aid in self.vertex_reservations.items():
            if t != timestep or aid == exclude_agent:
                continue
            if is_vertex_conflict(pos, (ry, rx)):
                return True
        return False

    def is_edge_reserved(
        self,
        from_pos: tuple[int, int],
        to_pos: tuple[int, int],
        timestep: int,
        exclude_agent: int
    ) -> bool:
        """
        辺衝突の予約チェック。
        予約済みエージェントの同一tickでの遷移と辺衝突するか判定。

        チェック内容:
        1. 予約済みの遷移 (other_from → other_to) に対して
           is_edge_conflict(from_pos, to_pos, other_from, other_to) を判定
        2. 加えて、予約済みエージェントが timestep+1 に to_pos 付近にいるか
           も頂点衝突として is_vertex_reserved で検査済み
        """
        for (t, oy1, ox1, oy2, ox2), aid in self.edge_reservations.items():
            if t != timestep or aid == exclude_agent:
                continue
            if is_edge_conflict(from_pos, to_pos, (oy1, ox1), (oy2, ox2)):
                return True
        return False
```

**注意:** `is_vertex_reserved()` と `is_edge_reserved()` は予約テーブル全体を線形探索する。エージェント数・tick数が大きい場合は、時刻をキーとした辞書にグループ化して高速化すること。

---

## 11. 充電スケジューラ (ChargingScheduler)

### 11.1 CS選択パラメータ（較正方針）

```python
# === CS選択の重みパラメータ ===
# 論文は式(4.57)の形式のみ定義し、数値を明示していない。
# 以下は較正パラメータとして扱う。
# Table 5.7/5.8 の目標値に近づくよう調整すること。

CS_SELECTION_PARAMS = {
    # 較正対象パラメータ（初期値）
    "w_end": 1.0,          # 総完了時間の重み
    "w_travel": 0.5,       # 移動時間の重み
    "w_queue": 1.0,        # 待ち行列時間の重み
    "t_buffer": 5,         # バッファ時間 [tick]
    "b_risk": 0.10 * B_MAX,  # リスク閾値 (= 200)
    "kappa": 10.0,         # リスク係数
    "k_candidates": 3,     # CS候補数上限

    # 緊急度係数 γ_urg（SoC区間 → 係数）
    "gamma_urg": {
        0.10: 5.0,   # SoC < 10%
        0.20: 3.0,   # 10% ≤ SoC < 20%
        0.30: 1.5,   # 20% ≤ SoC < 30%
        1.00: 0.0,   # 30% ≤ SoC
    },
}

# 較正方針:
# 1. 上記初期値でシミュレーション実行
# 2. Table 5.7/5.8 の目標値との乖離を確認
# 3. 主に w_queue, kappa, k_candidates を調整
# 4. 目標: proposed mean ≈ 93.4 (2 agents), ≈ 134.6 (3 agents)
```

### 11.2 CS選択（Jコスト最小化）

```python
class CSSelector:
    def __init__(self, params: dict = CS_SELECTION_PARAMS):
        self.params = params

    def select_cs(self, agent: Agent, system_state: SystemState) -> tuple[int, int] | None:
        k = self.params["k_candidates"]
        candidates = self._get_top_k_candidates(agent, system_state, k)
        if not candidates:
            return None
        best = min(candidates, key=lambda cs: self.compute_j_cost(agent, cs, system_state))
        return best

    def compute_j_cost(self, agent, cs_pos, system_state) -> float:
        p = self.params
        t_travel = self._estimate_travel_time(agent.position, cs_pos)
        t_wait = self._estimate_wait_time(cs_pos, system_state)
        b_arr = agent.battery - t_travel * DELTA_B_MOVE
        target_b = B_MAX * CHARGE_TARGET_THRESHOLD
        t_charge = compute_charge_time(max(0, b_arr), target_b)
        t_comp = t_travel + t_wait + t_charge + p["t_buffer"]
        gamma = self._urgency_coefficient(agent.battery)
        risk = self._risk_penalty(b_arr)
        return (p["w_end"] * t_comp +
                p["w_travel"] * t_travel +
                (p["w_queue"] + gamma) * t_wait +
                risk)

    def _urgency_coefficient(self, battery: float) -> float:
        soc = battery / B_MAX
        for threshold, coeff in sorted(self.params["gamma_urg"].items()):
            if soc < threshold:
                return coeff
        return 0.0

    def _risk_penalty(self, b_arr: float) -> float:
        b_risk = self.params["b_risk"]
        if b_arr < b_risk:
            return self.params["kappa"] * (b_risk - b_arr)
        return 0.0
```

### 11.3 最寄りCS選択（ベースライン）

```python
class NearestCSSelector:
    def select_cs(self, agent: Agent, system_state: SystemState) -> tuple[int, int] | None:
        """
        現在位置から最も近い到達可能CSを選択。
        到達可能 = BFS距離がINFでなく、かつ現在バッテリーで到達可能
        （b - d_cs * DELTA_B_MOVE >= 0）。
        待ち行列状況は考慮しない。
        """
```

### 11.4 充電挿入判定

```python
def should_insert_charging(
    agent: Agent,
    next_task: tuple[int, int],
    system_state: SystemState,
    min_goal_battery: float = B_GOAL_DEFAULT
) -> tuple[bool, tuple[int,int] | None]:
    """
    判定順序:
    1. ゴール到達不能 → 充電挿入
    2. ゴールからCSへの到達不能 → 充電挿入
    3. 安全残量: b_goal - e_{goal→cs} < B_safe → 充電挿入
    4. コスト比較: J(v*_cs) < J_direct → 充電挿入
    """
```

---

## 12. 緊急介入システム (EmergencySystem)

### 12.1 危険度判定（4.4.3節）

```python
class EmergencySystem:
    B_L4 = 0.02 * B_MAX  # 40 (即死: SoC 2%)
    B_L3 = 0.04 * B_MAX  # 80 (危険: SoC 4%)
    HISTORY_WINDOW = 10   # 減少率推定に使う履歴tick数

    def assess_danger_level(self, agent: Agent) -> int:
        """
        0: 正常, 1: 注意, 2: 警告, 3: 危険, 4: 即死
        レベル3以上で緊急介入発動。

        判定ロジック（優先順）:
        1. agent.battery <= B_L4 → レベル4
        2. agent.battery <= B_L3 → レベル3
        3. 履歴が HISTORY_WINDOW 未満 → SoC < 10% ならレベル3、
           SoC < 20% ならレベル2、それ以外レベル0
        4. CS上なのにバッテリーが減少している → レベル3
        5. 直近の減少率から枯渇予測時間を計算:
           rate = (history[-HISTORY_WINDOW].battery - agent.battery) / HISTORY_WINDOW
           if rate > 0:
               ticks_to_zero = agent.battery / rate
               if ticks_to_zero < 20 → レベル3
               if ticks_to_zero < 50 → レベル2
        6. それ以外 → レベル0
        """

    def execute_emergency_intervention(self, agent, system_state, planner):
        """
        緊急介入の処理順:
        1. 現在タスクを中断し、agent.interrupted_task に保存
        2. CSSelector.select_cs() で最適CSを選択
        3. goal_type="charging" の PlanningRequest を生成
        4. planner.plan() で経路再計画
        5. 計画成功 → 経路を割当
        6. 計画失敗 → Wait行動を割当（次tickで再試行）
        7. 充電完了後、interrupted_task を次のゴールとして再割当
        """
```

**注意:** 実験3では、予測的スケジューリングが十分に機能するため、緊急介入はほぼ発生しない。論文でも「予測的スケジューリングが十分に機能する環境では、緊急介入の発動頻度は低く抑えられることが期待される」と述べている。

---

## 13. タスク管理 (TaskManager)

### 13.1 タスク生成（乱数仕様の厳密定義）

```python
class TaskManager:
    def __init__(self, grid_map: GridMap, seed: int):
        self.rng = np.random.RandomState(seed)
        self.grid_map = grid_map
        self.valid_positions = self._precompute_valid_positions()

    def _precompute_valid_positions(self) -> list[tuple[int, int]]:
        """3×3のMoMoが配置可能な全位置をソート済みリストとして事前計算"""
        positions = []
        for y in range(1, self.grid_map.height - 1):
            for x in range(1, self.grid_map.width - 1):
                if is_valid_position(self.grid_map.grid, y, x):
                    positions.append((y, x))
        return sorted(positions)  # 再現性のためソート

    def generate_random_task(self, agent: Agent) -> tuple[int, int]:
        """
        乱数仕様:
        1. valid_positions からランダムにインデックスを選択
        2. 現在位置と一致する場合は再抽選
        3. np.random.RandomState を使用し、seed で完全再現

        注意: この方式は論文著者実装と異なる可能性がある。
        Table 5.7の再現性が低い場合、タスク生成方式を調整すること。
        """
        while True:
            idx = self.rng.randint(0, len(self.valid_positions))
            pos = self.valid_positions[idx]
            if pos != agent.position:
                return pos
```

### 13.2 完了タスク数のカウント規則

```python
# completed_tasks にカウントするもの:
#   current_goal_type == "task" のゴール到達のみ
#
# カウントしないもの:
#   current_goal_type == "charging" のCS到着
```

---

## 14. 統計収集 (StatisticsCollector)

### 14.1 収集する指標

```python
@dataclass
class SimulationStats:
    total_timesteps: int
    total_completed_tasks: int       # 全エージェント合計（chargingタスクを除く）
    per_agent_completed: dict[int, int]
    per_agent_charge_count: dict[int, int]
    collision_count: int
    energy_depletion_count: int
    cs_queue_lengths: dict[tuple, list[float]]  # CS座標 → 各tickのキュー長
    avg_cs_queue_length: float
    planning_times: list[float]
    expansion_counts: list[int]
    generation_counts: list[int]
    emergency_interventions: int
```

### 14.2 平均CSキュー長の定義

```python
def compute_avg_cs_queue_length(
    cs_queue_history: dict[tuple, list[int]],  # CS座標 → 各tickのキュー長
    total_ticks: int
) -> float:
    """
    平均CSキュー長 = (全CS × 全tickのキュー長の合計) / (CS数 × total_ticks)

    キュー長の定義:
    各tickにおいて、CSに向かって移動中（current_goal_type=="charging"
    かつ current_goal==cs_pos）のエージェント数。
    CS上で充電中のエージェントはキューに含めない。
    """
```

---

## 15. 実験再現仕様

### 15.1 実験1: 安全性フィルタとしての低レベル探索

**目的:** Low-Level探索がバッテリー制約を満たす経路のみ返すことを検証。

**共通設定:**
- 単一エージェント
- w_LL = 1.0（通常A*）
- L_SoC = 100（DELTA_SOC = 20.0）
- 初期回転レイヤ: z = 0
- **3×3占有チェック: 無効化**（実験1は低レベル探索の安全性フィルタ単体検証であり、論文でも安全性条件はS1: b(t)≥0 と S2: b_goal≥B_min の2条件のみ。占有チェックは実験3のマルチエージェント環境で機能する。）

**実験1のマップ定義:**
```python
# Case 1–6, 8 用: 直線マップ（3×3占有チェック無効のため1次元で十分）
EXP1_LINEAR_MAP_WIDTH = 50
EXP1_LINEAR_MAP_HEIGHT = 1

# Case 7 用: 2Dマップ（L字移動で回転が必要）
# 3×3占有チェック無効のため、最小限のサイズで良い
EXP1_2D_MAP_WIDTH = 20
EXP1_2D_MAP_HEIGHT = 20

# 実験1用の LowLevelPlanner 初期化時フラグ
EXP1_DISABLE_OCCUPANCY_CHECK = True  # is_valid_position() をバイパス
```

**全8ケースの完全定義:**

| Case | start | goal | CS位置 | 初期battery | B_min | 探索上限(Exp/Gen) | 期待状態 | 期待b_goal |
|---|---|---|---|---|---|---|---|---|
| 1: 到達不能（CSなし） | (0,0) | (0,49) | なし | 200.0 | 10.0 | 5000/10000 | 解なし | - |
| 2: 境界可到達（距離35） | (0,0) | (0,35) | なし | 190.0 | 15.0 | 5000/10000 | 解あり | 15.0 |
| 3: 境界可到達（距離40） | (0,0) | (0,40) | なし | 210.0 | 10.0 | 5000/10000 | 解あり | 10.0 |
| 4: 到達不能（距離45） | (0,0) | (0,45) | なし | 210.0 | 10.0 | 5000/10000 | 解なし | - |
| 5: 充電必須（CS経由） | (0,0) | (0,45) | [(0,20)] | 150.0 | 10.0 | 5000/10000 | 解あり | ≥10.0 |
| 6: ゴールがCS | (0,0) | (0,20) | [(0,20)] | 110.0 | 10.0 | 5000/10000 | 解あり | ≥10.0 |
| 7: L字経路（回転含む） | (0,0) | (15,15) | なし | 500.0 | 30.0 | 5000/10000 | 解あり | ≥30.0 |
| 8: 打ち切り | (0,0) | (0,45) | [(0,20)] | 150.0 | 10.0 | 5/19 | 打ち切り | - |

**補足:**
- Case 2: battery=190, 距離35, 消費=35×5=175, 残量=190-175=15=B_min（境界一致）
- Case 3: battery=210, 距離40, 消費=40×5=200, 残量=210-200=10=B_min（境界一致）
- Case 4: battery=210, 距離45, 消費=45×5=225>210（バッテリー不足）
- Case 7: 直線距離30（15+15）、回転1回（z=0→z=5）が必要

**検証項目:** Table 5.3 と一致する出力形式

```
Case | 状態 | S1 | S2 | b_goal | b_min | Charge | Rot | Exp/Gen | ms
```

### 15.2 実験2: SoC離散化粒度の影響

**目的:** 離散化粒度 L_SoC が解発見率と計算量に与える影響を評価。

**ケース定義:**
- Case A（境界条件）: 実験1 Case 2 と同一設定 (start=(0,0), goal=(0,35), battery=190, B_min=15, CSなし)
- Case B（充電必須）: 実験1 Case 5 と同一設定 (start=(0,0), goal=(0,45), CS=[(0,20)], battery=150, B_min=10)
- **3×3占有チェック: 実験1と同じく無効化**

**L_SoC別の探索上限:**

| L_SoC | DELTA_SOC | Case A 上限(Exp/Gen) | Case B 上限(Exp/Gen) |
|---|---|---|---|
| 10 | 200.0 | 5000/10000 | 50000/50000 |
| 20 | 100.0 | 5000/10000 | 100000/100000 |
| 40 | 50.0 | 5000/10000 | 100000/100000 |
| 100 | 20.0 | 5000/10000 | 5000/10000 |

**出力形式:**

```
# Table 5.4 相当
L_SoC | A状態 | A安全性 | B状態 | B安全性

# Table 5.5a 相当 (Case A)
L_SoC | Exp | Gen | ms

# Table 5.5b 相当 (Case B)
L_SoC | Exp | Gen | ms | Charge | Charge_nodes | Path_len | Limit
```

### 15.3 実験3: CS選択戦略の比較

**目的:** マルチエージェント環境での安定動作確認とCS選択戦略の比較。

**実験条件（完全定義）:**

| 項目 | 値 |
|---|---|
| マップ | Scenario1（2.4節で定義済み） |
| エージェント数 | 2, 3 |
| 初期位置 | SCENARIO1_INITIAL_POSITIONS_{2,3}_AGENTS |
| 初期回転レイヤ | 全エージェント z=0 |
| 初期バッテリー | 全エージェント B_MAX=2000 |
| 比較手法 | proposed（CSSelector）, nearest（NearestCSSelector） |
| 試行数 | 各条件20回（乱数シード 0, 1, 2, ..., 19） |
| 最大シミュレーション時間 | 3000 tick |
| L_SoC | 100 |
| w_HL | 1.5 |
| w_LL | 1.5 |
| B_GOAL (タスクゴール) | B_GOAL_DEFAULT = 10.0 |
| タスク生成 | TaskManager(seed=試行シード) |

**タスク生成の乱数管理:**
```python
# 各試行で1つの TaskManager を全エージェント共有
# エージェントがゴール完了するたびに generate_random_task() を1回呼ぶ
# 呼び出し順序: エージェントID昇順で処理
# これにより、同一シードで同一の乱数列が生成される
```

**目標値（論文 Table 5.7/5.8 — モード2 Table較正モードでの到達目標）:**

```
実験3の再現目標値:
  2 agents:
    proposed: mean=93.4, std=4.3, min=85, max=100
    nearest:  mean=89.8, std=4.9, min=83, max=100
    改善率: +4.0%
  3 agents:
    proposed: mean=134.6, std=4.4, min=126, max=141
    nearest:  mean=120.8, std=21.4, min=46, max=137
    改善率: +11.3%

平均CSキュー長:
  2 agents: proposed=0.07, nearest=0.19
  3 agents: proposed=0.15, nearest=0.67
```

**モード1（論文記述準拠）での検証基準:**
- 全試行で衝突=0、枯渇=0
- proposed > nearest（改善方向の一致）
- 3 agents で改善効果が 2 agents より大きい

**モード2（Table較正）での検証基準:**
- 上記に加え、平均完了タスク数が目標値の ±20% 以内（推奨 ±5%）

---

## 16. ファイル構成

```
momo_mapf/
├── main.py                          # エントリポイント
├── config.py                        # 全定数・較正パラメータ定義
│
├── models/
│   ├── __init__.py
│   ├── grid_map.py                  # GridMap クラス
│   ├── agent.py                     # Agent クラス
│   ├── battery.py                   # 充電レート、充電時間計算、離散化
│   ├── state.py                     # SearchState, SearchNode
│   └── occupancy.py                 # TimedOccupancy, ActionEntry, PlanResult
│
├── planners/
│   ├── __init__.py
│   ├── heuristic.py                 # EnergyAwareHeuristic
│   ├── low_level.py                 # LowLevelPlanner (Weighted A*)
│   ├── ecbs.py                      # ECBSPlanner (High-Level Search)
│   ├── conflict.py                  # Conflict, Constraint, ConflictDetector
│   └── reservation.py               # ReservationTable
│
├── scheduler/
│   ├── __init__.py
│   ├── charging_scheduler.py        # ChargingScheduler
│   ├── cs_selector.py               # CSSelector, NearestCSSelector
│   ├── emergency.py                 # EmergencySystem
│   └── task_manager.py              # TaskManager
│
├── simulation/
│   ├── __init__.py
│   ├── simulator.py                 # SimulationEngine（メインループ）
│   └── statistics.py                # StatisticsCollector
│
├── scenarios/
│   ├── __init__.py
│   └── scenario1.py                 # Scenario1マップ定義（推定座標）
│
├── experiments/
│   ├── __init__.py
│   ├── experiment1.py               # 実験1（8ケース完全定義）
│   ├── experiment2.py               # 実験2（4粒度 × 2ケース）
│   └── experiment3.py               # 実験3（完全定義）
│
└── utils/
    ├── __init__.py
    ├── bfs.py                       # BFS距離計算
    └── visualization.py             # 可視化ユーティリティ
```

---

## 17. エッジケースと注意事項

### 17.1 経路長の揃え

異なるエージェントの経路長が異なる場合、短い経路のエージェントは最終位置で待機（Wait）しているものとして扱う。衝突判定は最長経路の時刻まで行う。タイムラインにも待機tickを明示的に追加する。

### 17.2 タイムステップの整数化

C_MOVE=1, C_ROT_STEP=1, C_WAIT=1 はすべて整数。Charge時間も compute_charge_time() が整数(ceil)を返す。よって全時刻は整数tickとなる。完全方向転換（z=0↔z=5）は5回のRotateで5 tick。

### 17.3 充電中の占有と衝突

充電中のエージェントはCSセル上に留まる。タイムラインに全中間tickが登録され、予約テーブルにも反映される。他エージェントはこのCSに侵入できない。

### 17.4 初期配置

- 互いに衝突しない（式4.3を満たさない）位置に配置
- 初期回転レイヤは z=0
- 初期バッテリーは B_MAX（満充電）

### 17.5 デッドロック対処

ECBS探索のタイムアウトまたは展開上限で打ち切り。打ち切り時は待機(Wait)行動を割り当てて次tickに進む。

---

## 18. 検証基準

### 18.1 実験1 検証基準（両モード共通）

全8ケースの状態（解あり/解なし/打ち切り）とS1/S2判定が Table 5.3 と一致すること。3×3占有チェックは無効化されていることを確認。

### 18.2 実験2 検証基準（両モード共通）

L_SoC=100 で両ケース解あり、L_SoC≤40 で解なし/打ち切り傾向が一致すること。充電ノード生成数（Charge_nodes）が L_SoC=100 では正の値、粗粒度では 0 であること。

### 18.3 実験3 検証基準

**モード1（論文記述準拠）:**

| 基準 | 条件 | 必須度 |
|---|---|---|
| 安全性 | 全試行で衝突=0, 枯渇=0 | 必須 |
| 改善方向 | proposed mean > nearest mean（両エージェント数） | 必須 |
| スケール効果 | 3 agents の改善率 > 2 agents の改善率 | 必須 |
| 安定性 | proposed std < nearest std（特に 3 agents） | 必須 |
| CS分散 | proposed の平均CSキュー長 < nearest | 必須 |

**モード2（Table較正）— モード1の基準に加えて:**

| 基準 | 条件 | 必須度 |
|---|---|---|
| 数値近似 | 平均完了タスク数が目標値の ±20% 以内 | 推奨 |
| 数値一致 | 平均完了タスク数が目標値の ±5% 以内 | 理想 |

モード2で数値一致が達成できない場合の調整順序:
1. Scenario1 の障害物・CS座標を微調整
2. CS選択パラメータ（CS_SELECTION_PARAMS）を較正
3. タスク生成の乱数方式を変更
4. w_HL, w_LL の値を調整
