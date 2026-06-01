# sprint-speed-imu

WT9011DCL または類似 IMU の CSV ログから、陸上短距離走の推定走速度曲線を生成する Python CLI です。

横軸を距離 `m`、縦軸を推定速度 `m/s` としたグラフ、解析 CSV、実行条件、QC、警告を出力します。

> [!IMPORTANT]
> 本ツールが出力する速度、距離、最高速度地点は IMU ログからの推定値です。公式記録、光電管、フォトフィニッシュ、認定区間速度に相当する値ではありません。練習時の傾向確認、相対比較、ログ品質確認を主目的にしてください。

## 主な機能

- WT9011DCL / 類似 IMU の CSV ログを読み込み
- PCA 方式による走行方向推定
- 姿勢角 / クォータニオンを使った attitude 方式
- 方位角 / 磁気センサーを使った heading 方式
- 既知距離による manual 距離アンカー
- IMU 積分のみの auto 距離推定
- heading 方式では推定 2D 軌跡を出力
- Savitzky-Golay、Butterworth、Kalman / RTS smoother による平滑化
- `summary.json` への QC、confidence、warning、実行条件の出力
- `compare` による PCA / attitude の比較
- `synthetic` による検証用 CSV 生成

## 動作要件

| 項目 | 要件 |
|---|---|
| OS | macOS を主対象 |
| Python | Python 3.11 以上 |
| 入力 | 取得済み IMU CSV |
| 依存 | `numpy`, `pandas`, `scipy`, `matplotlib` |
| テスト | `pytest` |
| リアルタイム BLE | 非対応 |
| PDF レポート | 非対応 |

Linux でも動く可能性はありますが、現時点の主対象は macOS です。

## インストール

ソースからインストールします。

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -e .
```

開発・テスト込みで入れる場合:

```bash
pip install -e ".[test]"
```

動作確認:

```bash
sprint-speed-imu --help
```

## クイックスタート

まずは合成データで動作確認します。

```bash
sprint-speed-imu synthetic \
  --distance-m 50 \
  --duration-s 6.5 \
  --sample-rate-hz 100 \
  --yaw-deg 30 \
  --output examples/synthetic_run.csv \
  --overwrite
```

PCA 方式 + 既知距離で解析します。

```bash
sprint-speed-imu \
  --input examples/synthetic_run.csv \
  --method pca \
  --distance-source manual \
  --distance-m 50 \
  --output-dir out/test_pca \
  --overwrite
```

出力:

```text
out/test_pca/
  speed_curve.png
  speed_curve.csv
  summary.json
  run_config.json
  warnings.txt
```

## 推奨ワークフロー

実データでは以下を推奨します。

1. センサーを腰、仙骨付近、背中下部にしっかり固定する
2. 走行距離を実測し、`--distance-source manual --distance-m <m>` を使う
3. 可能なら `--duration-s` または `--end-mode manual --end-time` を指定する
4. まず `--method pca` で解析する
5. 姿勢角またはクォータニオンが信頼できる場合だけ `--method attitude` と比較する
6. 距離不明・非直線の腰装着ログでは `--method heading --distance-source auto` を参考解析として使う
7. `summary.json` の `warnings`, `confidence`, `qc` を確認する

最良:

```bash
sprint-speed-imu \
  --input data/run_50m.csv \
  --method pca \
  --distance-source manual \
  --distance-m 50 \
  --duration-s 6.50 \
  --acc-unit g \
  --output-dir out/run_50m_pca
```

CSV が走行開始からゴールまでに切られている場合:

```bash
sprint-speed-imu \
  --input data/run_50m_trimmed.csv \
  --method pca \
  --distance-source manual \
  --distance-m 50 \
  --end-mode data-end \
  --output-dir out/run_50m_pca
```

PCA / attitude を比較:

```bash
sprint-speed-imu compare \
  --input data/run_50m.csv \
  --distance-source manual \
  --distance-m 50 \
  --duration-s 6.50 \
  --output-dir out/run_50m_compare
```

## 入力 CSV

標準列:

```csv
time_s,ax,ay,az,gx,gy,gz,roll,pitch,yaw,qw,qx,qy,qz,hx,hy,hz
```

### 必須列

| 方式 | 必須列 |
|---|---|
| `pca` | `time_s`, `ax`, `ay`, `az` |
| `attitude` | `time_s`, `ax`, `ay`, `az` と `roll,pitch,yaw` または `qw,qx,qy,qz` |
| `heading` | `time_s`, `ax`, `ay`, `az` と `yaw`、`qw,qx,qy,qz`、または `hx,hy,hz` |

`time_s` が無い場合は `--sample-rate-hz` から時刻列を生成できます。
カンマ区切り CSV とタブ区切り TSV は自動判定されます。
`time` 列が ISO 日時の場合は、先頭行からの経過秒に変換されます。

```bash
sprint-speed-imu \
  --input data/run_without_time.csv \
  --sample-rate-hz 100 \
  --method pca \
  --distance-source manual \
  --distance-m 50 \
  --output-dir out/run_without_time
```

### 単位

| オプション | 既定 | 値 |
|---|---:|---|
| `--acc-unit` | `g` | `g`, `mps2` |
| `--gyro-unit` | `dps` | `dps`, `radps` |
| `--angle-unit` | `deg` | `deg`, `rad` |

`--acc-unit g` の場合、加速度は内部で `m/s^2` に変換されます。

### 列名マッピング

WitMotion など、CSV の列名が標準列と異なる場合は `--column-map` を使います。
WT901BLE 形式の `AccX(g)`, `AsX(°/s)`, `AngleX(°)`, `Q0()` などの単位付き列名は、
追加マッピングなしで読み込めます。

例:

```json
{
  "time_s": "Time",
  "ax": "AccX",
  "ay": "AccY",
  "az": "AccZ",
  "gx": "GyroX",
  "gy": "GyroY",
  "gz": "GyroZ",
  "roll": "AngleX",
  "pitch": "AngleY",
  "yaw": "AngleZ",
  "hx": "HX",
  "hy": "HY",
  "hz": "HZ"
}
```

実行例:

```bash
sprint-speed-imu \
  --input data/witmotion.csv \
  --column-map examples/column_map_witmotion.json \
  --method pca \
  --distance-source manual \
  --distance-m 50 \
  --output-dir out/witmotion_pca
```

## 解析方式

### `--method pca`

加速度から重力方向を推定し、水平成分の主成分から走行方向を推定します。

推奨条件:

- 直線走である
- センサーが体幹、腰、仙骨付近に固定されている
- yaw や姿勢角の信頼性が低い
- 角度列が無い

主な QC:

- `pca_explained_variance_ratio`
- `pc1_vector`
- `baseline_gravity_norm_mps2`
- `horizontal_energy_ratio`

### `--method attitude`

クォータニオン、または `roll,pitch,yaw` からセンサー座標の加速度をワールド座標へ変換し、指定したコース方向に射影します。

推奨条件:

- クォータニオンまたは姿勢角が安定している
- yaw が大きく乱れていない
- 直線走で進行方向が概ね分かっている

関連オプション:

```bash
--course-yaw-deg 0
--euler-order xyz
```

### `--method heading`

`yaw`、クォータニオンから得たセンサー前方軸、または `hx,hy,hz` の磁気センサーから時々刻々の方位を推定し、重力除去後の水平加速度をその方位へ射影します。

推奨条件:

- 腰、仙骨、背中下部に固定されている
- 距離が分からず、かつ直線とは限らない移動を参考解析したい
- `AngleZ` またはクォータニオンが概ね安定している
- 磁気センサー周辺に強い磁気ノイズ源が少ない

実行例:

```bash
sprint-speed-imu \
  --input data/waist_walk.tsv \
  --method heading \
  --distance-source auto \
  --smoothing-method kalman \
  --output-dir out/waist_heading
```

関連オプション:

```bash
--heading-source auto          # auto, yaw, quaternion, magnetometer
--heading-offset-deg 0         # センサー前方と実際の腰向きの補正角
```

`heading` は歩幅や manual 距離を使いません。そのため、進行方向の追従には有用ですが、速度と距離は加速度積分のドリフトを含む低信頼の推定値です。

## 距離モード

### `--distance-source manual`

既知距離を強いアンカーとして使います。実運用ではこのモードを推奨します。

```bash
--distance-source manual --distance-m 50
```

`manual` では、速度曲線の最終距離が `--distance-m` に合うよう補正されます。

### `--distance-source auto`

IMU 積分のみで距離を推定します。これは低信頼・実験用です。

```bash
--distance-source auto
```

`auto` の結果はドリフトの影響を大きく受けます。`pca` / `attitude` の速度評価では `manual` を使ってください。
距離不明の `heading` 解析では、低信頼の参考値として扱ってください。

## 速度補正モード

`--correction-mode` で速度推定の補正方法を選べます。

| 値 | 説明 |
|---|---|
| `mean-speed-shape` | 既定。速度形状を既知距離へスケール |
| `scale` | 非負化した raw 速度を距離へスケール |
| `bias` | 加速度バイアス補正後に距離へ整合 |
| `raw-integration` | raw 積分形状を使い、manual 時は最終距離へスケール |

## 平滑化

`--smoothing-method` は以下を指定できます。

| 値 | 説明 |
|---|---|
| `savgol` | 既定。Savitzky-Golay filter |
| `butter` | Butterworth low-pass filter |
| `kalman` | 1D Kalman filter + RTS smoother |
| `none` | 平滑化しない |

Kalman の例:

```bash
sprint-speed-imu \
  --input data/run_50m.csv \
  --method pca \
  --distance-source manual \
  --distance-m 50 \
  --smoothing-method kalman \
  --kalman-process-noise 0.05 \
  --kalman-measurement-noise 0.5 \
  --output-dir out/run_50m_kalman
```

`--kalman-process-noise` を小さくする、または `--kalman-measurement-noise` を大きくすると平滑化は強くなります。ただし加速ピークを丸める可能性があります。`summary.json` の `velocity_diagnostics.forward_accel_peak_retention_ratio` を確認してください。

## CLI リファレンス

通常解析:

```bash
sprint-speed-imu [run] --input INPUT --output-dir OUTPUT_DIR --method {pca,attitude,heading} --distance-source {manual,auto}
```

主要オプション:

| オプション | 説明 |
|---|---|
| `--input` | 入力 CSV |
| `--output-dir` | 出力ディレクトリ |
| `--method` | `pca`, `attitude`, `heading` |
| `--distance-source` | `manual` または `auto` |
| `--distance-m` | manual 距離 `m` |
| `--start-mode` | `auto` または `manual` |
| `--start-time` | manual start 秒 |
| `--end-mode` | `data-end` または `manual` |
| `--end-time` | manual end 秒 |
| `--duration-s` | start からの走行時間秒 |
| `--distance-bin-m` | speed_curve.csv の距離刻み |
| `--smooth` | `true` または `false` |
| `--smoothing-method` | `savgol`, `butter`, `kalman`, `none` |
| `--heading-source` | `auto`, `yaw`, `quaternion`, `magnetometer` |
| `--heading-offset-deg` | heading 方式の方位補正角 |
| `--pca-window` | `run`, `first-2s`, `first-half` |
| `--strict` | warning がある場合に失敗 |
| `--overwrite` | 出力先を上書き |
| `--debug` | 中間 CSV を出力 |

比較:

```bash
sprint-speed-imu compare \
  --input data/run.csv \
  --distance-source manual \
  --distance-m 50 \
  --output-dir out/compare
```

合成データ:

```bash
sprint-speed-imu synthetic \
  --distance-m 50 \
  --duration-s 6.5 \
  --sample-rate-hz 100 \
  --yaw-deg 30 \
  --output examples/synthetic_run.csv
```

## 出力ファイル

| ファイル | 内容 |
|---|---|
| `speed_curve.png` | 距離 - 推定速度グラフ |
| `speed_curve.csv` | 距離ビンごとの推定速度 |
| `summary.json` | 結果、QC、confidence、warning、仮定 |
| `run_config.json` | 実行時設定 |
| `warnings.txt` | warning の一覧 |
| `debug_intermediate.csv` | `--debug` 指定時の中間信号 |
| `trajectory.csv` | `heading` 方式時の推定 2D 軌跡 |
| `trajectory.png` | `heading` 方式時の推定 2D 軌跡グラフ |

### `speed_curve.csv`

主な列:

| 列 | 説明 |
|---|---|
| `distance_m` | 距離 `m` |
| `time_s` | 対応する時刻秒 |
| `estimated_speed_mps` | 推定速度 `m/s` |
| `a_forward_mps2` | 速度推定に使った前方加速度 |
| `confidence_flag` | overall confidence |

### `summary.json`

確認すべき主なフィールド:

| フィールド | 説明 |
|---|---|
| `run` | 解析条件 |
| `results.average_speed_mps` | 平均速度 |
| `results.max_estimated_speed_mps` | 最大推定速度 |
| `confidence.overall` | 全体信頼度 |
| `warnings` | 注意事項 |
| `qc.pca_explained_variance_ratio` | PCA 方向推定の寄与率 |
| `qc.yaw_std_deg` | yaw の安定性 |
| `qc.heading_source` | heading 方式で使った方位ソース |
| `qc.heading_lateral_energy_ratio` | 方位方向に対する横方向加速度エネルギー比 |
| `qc.magnetic_norm_cv` | 磁気センサー強度の変動係数 |
| `qc.manual_distance_correction_ratio` | 距離補正比 |
| `qc.acceleration_saturation_warning` | 加速度飽和疑い |
| `velocity_diagnostics.forward_accel_peak_retention_ratio` | 平滑化後の加速度ピーク保持率 |

## Confidence の目安

| 値 | 目安 |
|---|---|
| `medium` | manual 距離、時間条件、QC が概ね良好 |
| `low_to_medium` | manual 距離だが終了時刻などに不確実性あり |
| `low` | auto 距離、QC 低下、警告条件あり |

`medium` でも実測速度を意味しません。あくまで推定曲線としての相対的な信頼度です。

## 運用要件

精度と再現性のため、以下を守ってください。

- `pca` / `attitude` は直線走で使う
- 距離不明・非直線の腰装着ログは `heading` を参考解析として使う
- センサーを腰、仙骨、背中下部に固定する
- ポケット、緩いポーチ、揺れる固定具は避ける
- 初期運用ではシューズ装着を避ける
- 既知距離を入力する
- 既知距離を使わない `heading` / `auto` では絶対速度を過信しない
- 可能なら走行時間、終了時刻、またはトリム済み CSV を使う
- `summary.json` の warning を確認する
- 異なる選手、装着位置、センサー設定を直接比較しない

## 非対応・制限

- BLE リアルタイム取得は非対応
- 公式タイムや公認速度の算出は非対応
- `pca` / `attitude` のカーブ走、急な方向転換、多人数ログは想定外
- IMU 単体の auto 距離推定は低信頼
- heading 方式でも加速度積分ドリフトにより絶対速度と距離は低信頼
- 磁気センサーは屋内の金属、電子機器、センサー校正の影響を受ける
- 靴装着では加速度飽和の可能性が高い
- センサー座標系や Euler 順序は機種・設定に依存するため実データで確認が必要

## トラブルシュート

| 症状 | 主な原因 | 対策 |
|---|---|---|
| 速度が極端に高い | 単位、距離、時間範囲の誤り | `--acc-unit`, `--distance-m`, `--duration-s` を確認 |
| 速度曲線が暴れる | 固定不良、ノイズ、方向推定失敗 | 固定を見直し、`--smoothing-method kalman` や `pca-window` を試す |
| PCA confidence が低い | 主方向が明確でない | `--pca-window first-2s`、固定位置、CSV 範囲を確認 |
| attitude が失敗する | 姿勢列が無効、Euler 順序不一致 | `--method pca` を使う、`--euler-order` を確認 |
| heading の軌跡が曲がりすぎる | yaw 不安定、磁気外乱、センサー前方ズレ | `--heading-source`, `--heading-offset-deg`, 固定位置を確認 |
| auto 距離が変 | IMU 積分ドリフト | `--distance-source manual` を使う |
| strict で失敗する | warning が出ている | `warnings.txt` と `summary.json` を確認 |

## テスト

```bash
pip install -e ".[test]"
pytest
```

合成データを使った smoke test:

```bash
sprint-speed-imu synthetic \
  --distance-m 50 \
  --duration-s 6.5 \
  --output examples/synthetic_run.csv \
  --overwrite

sprint-speed-imu \
  --input examples/synthetic_run.csv \
  --method pca \
  --distance-source manual \
  --distance-m 50 \
  --output-dir out/test_pca \
  --overwrite
```

## ライセンス

このプロジェクトは MIT License で公開されています。詳細は [LICENSE](LICENSE) を確認してください。

本ツールの推定結果は保証された計測値ではありません。利用者は `summary.json` の warning と QC を確認し、自己責任で解釈してください。
