# AGD保持型・効率的大受容野 depthwise conv — 実装と検証レポート

対象: UniConvNet-T U-Net の RFA(ConvMod) 大カーネル depthwise(a1=7×7 / a2=9×9 / a3=11×11)。
骨シンチ13クラス segmentation(val n=955, seed42)。

## 1. 動機

先行検討で「単一 **dilated** conv で depthwise を置換すると疎サンプリングで ERF のガウス性
(AGD=漸近ガウス分布)が崩れる」ことが判明。そこで dilated を使わず、**dense/ガウスの意味論を
保ったまま大カーネルを効率化する 2 機構**を実装し、(1) AGD が保たれるか、(2) 計算コストが
下がるか、(3) 骨シンチで精度が回帰しないか、を検証した。

## 2. 実装した 2 機構

| 機構 | 実装 | 要点 |
|---|---|---|
| A: 分離 | `src/models/separable_dw.py` `SeparableDWConv` | K×K を R 本の 1×K・K×1 の和に分解(K²→2RK, `--separable-rank R` 既定1)。密カーネルを rank-R SVD で分離初期化。R を上げると非分離構造=境界を回収(§12) |
| B: スペクトル | `src/models/spectral_dw.py` `SpectralDW` | 周波数ガウス低域通過 + 動的スペクトル切り出し(η=α/σ)。σ 大で帯域を詰め安く |
| C: 多ガウス混合 | `src/models/spectral_mixture_dw.py` `SpectralMixtureDW` | 周波数包絡を **K ガウス混合** Σw_k·exp(−2π²σ_k²f²) に。ERF を集約ガウス(AGD)へ厳密化。分離 rank-K 構築 + 振幅考慮切り出しで単一ガウス並みのコスト(§10) |

統合: `ConvMod` に `dw_mode ∈ {dense, separable, spectral, spectral_mix}` を追加(既定 dense は現状維持)。
評価系は `build_model_from_checkpoint` で dw_mode を自動判別(機構C は log_sigma が (C,K) 2 次元でも判別)。

## 3. 単層 AGD(学習不要, `src/agd_probe.py`)

同一 RF σ=6 に揃えた実効カーネルの excess kurtosis / FFT sidelobe:

| 機構 | excess kurtosis | FFT sidelobe |
|---|---|---|
| dilated(d=4) | −0.48 | **1.00**(複製ローブ=AGD崩壊) |
| separable | −0.15 | 0.0012(= dense と厳密一致) |
| spectral(無切り出し) | +0.00 | 0.000 |

→ **設計は正しい**: dilated は疎サンプリングでエイリアシング、機構A/B は単層でガウスを保つ。
切り出しは α=2(η=0.33, cost≈11%)で L2 誤差 1e-8 の実質無損失。

## 4. コスト(`src/benchmark.py`, 256px)

| model | params | img/s(実速度) | RFA省コスト比(理論) |
|---|---|---|---|
| baseline(dense) | 36.54 M | 98.7 | 1.0 |
| separable | 35.86 M | **113.7 (+15%)** | 4.95 |
| spectral(pure) | 36.56 M | 76.3 (−23%) | 14.0 |
| spectral(learned σ) | 36.56 M | **60.2 (−39%)** | 22.5(虚構) |

→ **separable のみ実速度も改善**。spectral は FFT オーバーヘッドで壁時計は悪化。
RFA省コスト比は MAC 上の理論値で、spectral では img/s と逆行(=絵に描いた餅)。

## 5. 精度・境界・AGD(骨シンチ実データ)

3 機構すべて **Dice は baseline と中立**(overall 0.939, worst10% 0.898、全同値)。
局所タスクゆえ RF は精度に無関係、という既存結論と整合。境界(hd95/assd)/AGD(層別kurtosis)で差:

- **separable**: 速い+AGD保持(kurtosis ≈ baseline)+RF保持。ただし rank-1 制約で
  **境界が有意に悪化**(worst10% assd 1.16→2.31=2倍, p4e-7)。滑らかさと境界鋭さの
  トレードオフが rank-1 の裏表として現れる。
- **spectral(pure)**: worst 群では精度も境界も中立。ただし遅い/全モデルERF kurtosis 崩壊
  (encoder_stage0 kurt 221)/循環畳み込みで稀な境界破綻(1枚 hd95 57)。
- **spectral(learned σ)**: σ高lr で σ が実学習・per-ch 分化(92-95%動く)。RF が baseline へ
  回復、AGD も stage2/3 で回復(kurt 7/3)、stage0/1 は改善だが残る(84/33)。**だが σ 分化で
  truncation が死んで最遅、破綻は場所が移るだけ**。

## 6. σ の学習可能性(`src/visualize_sigma.py`)

lr1e-4 では σ が完全凍結(実測 |Δlog|~0.005 << σ2倍に必要な 0.69)。
`--spectral-sigma-lr 1e-2`(log_sigma 専用高lr・WD0)で **σ は実学習し per-ch 分化**する。
→ σ が動く条件: ①σが勾配を受ける経路(pure-spec か gamma>0)+ ②σ専用高lr。
identity init(local+gamma=0)は spec をゲートし σ 勾配=0。

## 7. 結論

- **設計は正しい**(単層でガウス保持・切り出しは無損失・σは適切なlrで学習/分化)。
- **骨シンチは局所タスクゆえ精度利得ゼロ**は 3 機構とも不変。
- 効率+AGD+精度の三拍子を同時達成する機構は本タスクには無い:
  separable=速いが境界劣化、spectral=中立だが遅い/破綻/AGD崩壊(pure)。
- **貢献の軸**: (a) 評価基盤(全指標×worst×境界)が「Dice横並びでも assd 2倍」を暴く、
  (b) 機構の振る舞い自体(σ学習・AGD制御・rank-1のトレードオフ)を定量化。

## 8. spectral の改善(commit 8443def)

learned_sigma の 2 弱点にノブを追加(既定オフ=後方互換):
- `--spectral-pad-factor`(reflect パディング): 循環畳み込みの巻き込み破綻を抑制
  (単体: 反対角漏れ 2.3e-3→7.1e-8)。
- `--spectral-crop-quantile`(σ_ref を分位点化): σ 分化しても truncation を維持
  (単体: min keep100% vs q0.2 keep6%)。

推奨再学習(境界を守り σ を学習し破綻せず省コストを維持):
```
CUDA_VISIBLE_DEVICES=0 python3 src/train.py --dw-mode spectral \
  --spectral-use-local-branch --spectral-init-gamma 0.5 --spectral-sigma-lr 1e-2 \
  --spectral-init-sigma 8 6 4 2 --spectral-alpha 2 \
  --spectral-pad-factor 0.25 --spectral-crop-quantile 0.1 --batch-size 16 --tag spec_v2
```

## 10. 機構C: 多ガウス混合の周波数領域 depthwise(`SpectralMixtureDW`)

### 10.1 動機 — 単一ガウスは AGD の"甘い"近似

`erf_gmm_fit.py` の観察: RFA の ERF は**単一ガウスでは乗り切らず、N(=3)ガウスの和で綺麗に乗る**
(= 集約ガウス aggregated Gaussian)。RFA は a1/a2/a3(k7/9/11)の 3 スケールを集約するので、
受容野は本質的に複数スケールのガウスの重ね合わせ。機構B の単一ガウス包絡はこの構造を 1 本に
潰した近似。そこで**周波数包絡そのものを K ガウス混合**にする:

    Ŵ_c(f) = Σ_{k=1}^K w_{c,k}·exp(−2π²σ_{c,k}²|f|²)

ガウスのフーリエ双対より、これは空間核が **ガウス混合** w_c(r)=Σ_k w_{c,k}·N(r;0,σ_{c,k}²I)
であることと厳密に等価 → ERF が定義から集約ガウス = **AGD をネイティブかつ単一ガウスより高表現力で保証**。
K=1 は機構B(単一ガウス)に厳密一致(真の一般化)。

### 10.2 効率的計算の 3 つの要(多ガウスでもコストを単一ガウス並みに保つ)

1. **分離 rank-K 構築**: 等方ガウスは軸方向に分離 exp(−2π²σ²(f_y²+f_x²))=exp(·f_y²)·exp(·f_x²)。
   K 混合包絡は **rank-K** テンソル = K 外積の和。1D 指数 G_y(C,K,H'), G_x(C,K,W') から einsum で
   (C,H',W') を組む → exp 評価は K·C·(H'+W') 回(素朴 2D の K·C·H'·W' より軽い)、バッチ複素乗算は 1 回。
2. **振幅考慮の動的切り出し**: 必要帯域は「まだ重みを持つ最もシャープな成分」で決まる。
   σ_ref = min{σ_{c,k}: w_{c,k}≥τ}、η=clamp(α/σ_ref, η_min, 1)。振幅の薄いシャープ成分は帯域予算から
   外れる → 単一ガウスの η=α/σ より厳密にタイト(AGD 忠実度は保持)。
3. **DC 保存の凸混合**: w=softmax(Σ_k w=1, w≥0)、各ガウス DC 利得=1 → 包絡は f=0 で 1 →
   平均を保つ多スケール低域通過。残差 local+gamma·spectral に安定に足せる(gamma=0 初期で元 conv と厳密一致)。

パラメータ: `weight/bias`(a*.2 と同形・同キー→事前学習転移)、`log_sigma`(C,K)、`mix_logits`(C,K)、`gamma`(C)。
新規パラメータは strict=False 吸収。`log_sigma` 命名により train.py の σ 専用 param group(高lr/WD0)へ自動編入。

### 10.3 設計検証(CPU, `smoke_test.py` + 単体数値チェック)

| 主張 | 検証 | 結果 |
|---|---|---|
| 分離 rank-K == 素朴 2D 混合 | ランダム σ/w で包絡を両法で構築し比較 | max abs err **5.96e-8**(厳密) |
| 凸混合の DC 保存 | 包絡 f=0 の値 | **1.000000** |
| K=1 == SpectralDW | 局所枝/σ/gamma を揃え forward 比較 | max err **1.19e-7** |
| ERF = 集約ガウス | パルス応答が非負・二次モーメント σ | 非負(min −1.6e-10)、実測 σ=7.94 == `effective_sigma()`=7.94 |

`smoke_test.py`: 全 dw_mode(dense/spectral/spectral_mix/separable)で forward・勾配・RF 拡大を PASS。
FLOPs@256: dense 19.58G / spectral 19.26G / **spectral_mix 18.29G** / separable 18.42G
(mix は多スケール初期化 σ∈[init/2, init·2] で振幅考慮切り出しが効き spectral より安い)。

### 10.4 推奨学習レシピ(骨シンチ)

```
CUDA_VISIBLE_DEVICES=0 python3 src/train.py --dw-mode spectral_mix \
  --spectral-num-gaussians 3 --spectral-use-local-branch --spectral-init-gamma 0.5 \
  --spectral-sigma-lr 1e-2 --spectral-init-sigma 8 6 4 2 --spectral-alpha 2 \
  --spectral-pad-factor 0.25 --spectral-crop-quantile 0.1 --batch-size 16 --tag mix_v1
```

狙い: 機構B spec_v2 と同条件で **単一ガウス→多ガウス混合**の差分を見る。仮説は
「ERF の集約ガウス性が構造的に保たれ、機構B(learned σ)で stage0/1 に残った kurtosis 崩壊が
緩和」。→ 実測結果は §10.5(仮説は**部分棄却**: 集約ガウス性は保つが kurtosis は改善せず)。

### 10.5 骨シンチ実測結果(run `mix_v1`, val n=955, seed42)

3機構(dense=対照 / spectral=機構B learned σ / mix=機構C K=3)を同一 val で比較
(`erf_sigma_table` / `erf_gmm_fit` / `compare_models` / `benchmark`)。

| 指標 | dense | spectral(B) | mix(C) |
|---|---|---|---|
| best_bone_dice | 0.9413 | 0.9410 | 0.9410 |
| dice overall / worst10% | ~0.941 / ~0.90 | 0.941 / 0.90 | 0.939 / 0.896 |
| **encoder_stage0 kurtosis** | **4.6** | 78.0 | **93.5** |
| encoder_stage3 kurtosis | 1.3 | 2.7 | 2.5 |
| GMM R² stage0 (1G→3G) | 0.910→0.931 | 0.906→0.930 | 0.899→0.916 |
| GMM R² stage3 (1G→3G) | 0.838→**0.934** | 0.853→0.896 | 0.829→0.903 |
| σ_moment (stage0 / stage3) | 8.3 / 80.3 | 7.4 / 75.0 | 7.2 / 73.8 |
| FLOPs@256 / img-s | 19.6G / 98.7 | 19.3G / 60.2 | **18.3G / 58.3** |

読み取れる結論:

1. **AGD の正しい物差し(GMM)では機構C成立**: 3ガウス和が単一ガウスを全機構で明確に上回る
   (stage0 0.90→0.93, stage3 mix 0.83→0.90)= ERF は集約ガウス。設計前提を実データで裏付け。
2. **kurtosis(単一ガウス基準)では機構C改善せず、むしろ悪化**: dense の stage0 は kurt 4.6 と
   ほぼガウスなのに spectral 78 / mix 93。「鋭い local conv + 広い spectral ハロー」の合成で中心が
   尖り、**混合の広い裾がむしろ尖度を上げる**(mix 93 > spectral 78)。ただし GMM R²=0.92 で 3ガウスに
   乗る=エイリアシング崩壊ではなく正当な narrow+wide 混合。report が言う通り excess-kurtosis は
   “物差し違い”。→ **機構C は AGD を機構B より改善しない。dense が最も浅層AGDが綺麗**、という対照が確定。
3. **精度中立・効率は設計通り**: dice は3機構横並び(局所タスク)。mix の FLOPs(18.3G)は spectral より
   低く実速度は同等(58 vs 60 img/s)= K=3 でも単一ガウス並みのコスト(分離 rank-K が効いている)。

要注意(継承した弱点): worst サンプル 1 枚(`20070509_0070000065_A`)で mix のみ **assd 56 / hd95 74** の
壊滅的破綻(dense は assd 3.3)。pad_factor=0.25 でも 1/955 で FFT 循環畳み込みの破綻が残り、これが
worst10% assd(1.73)を押し上げる(§5 の spectral 稀な境界破綻を機構Cも継承)。

総括: 機構C は**正しく実装され効率的で、集約ガウスをネイティブ保持する**が、**骨シンチでは AGD も
精度も機構B を超えない**(局所タスクゆえ)。貢献は (a) 多ガウス FFT の効率設計そのもの、
(b)「dense が浅層 AGD の最良保持者」という対照の定量化、(c) kurtosis vs GMM の物差し問題の実証。

## 12. 機構A の rank-k 拡張(境界を取り戻す・骨シンチ本命の一手)

### 12.1 動機 — rank-1 の唯一の欠点は capacity 損失

§5/§7 で確定した通り、機構A(separable rank-1)は**速い(+15% 実速度)+ AGD/RF を保持**するが、
**Dice/境界すべてで baseline より有意に悪化**する(dice overall Δ−0.0045 p5e-77、worst10% assd
1.16→2.31=2倍 p4e-7、precision↓+recall↑=過分割/境界が甘い)。原因は一意に**事前学習カーネルの
rank-1 近似**(rank-1 energy ≈0.80 = 20% の非分離構造=境界の鋭さを捨てている)。rank-1 は滑らかで
AGD/kurtosis を保つが、同時に鋭い非分離構造を表現できない——「滑らかさ(AGD)」と「境界の鋭さ」が
rank-1 の表裏のトレードオフ。

**本物の事前学習カーネルで前提を実測**(`uniconvnet_t_1k_224_ema.pth` の a1/a2/a3 depthwise
72層=8928 チャネルを per-channel SVD): 平均 captured energy は **rank1=0.802 → rank2=0.929
(+0.127) → rank3=0.962**。カーネル別では a1(k7)0.839→0.955、a2(k9)0.792→0.929、
**a3(k11, 最も非分離)0.796→0.920**。→ **rank-2 で 80%→93% を回収**でき、境界劣化を戻せる
余地が定量的にある(rank-1 の 0.80 は 20% を捨てているが、rank-2 でその 6割強を取り戻す)。

### 12.2 設計 — R 本の分離の和(rank-R)

K×K カーネルを **R 本の 1×K・K×1 の和**で近似する:

    y = Σ_{r=1}^R DWConv_{K×1}^{(r)}( DWConv_{1×K}^{(r)}(x) )

合成カーネルは K2[c,i,j] = Σ_r weight_v[c,r,i]·weight_h[c,r,j](= per-channel の rank-R 行列)。
事前学習の密カーネルを**チャネルごと rank-R SVD**(Eckart–Young 最良近似)で分離初期化する:
weight_v[c,r]=√s_r·u_r、weight_h[c,r]=√s_r·v_r。**R=1 で従来の rank-1、R=K で密カーネルに厳密一致**
(分離近似ではなくなる)→ **rank R が「精度-効率-AGD」を連続制御するノブ**になる。

- コスト: タップ 2RK(密 K²)。K=11 で R=1→22(5.5x減)、**R=2→44(2.75x減)**、R=3→66(1.83x減)。
- 実装: `weight_h`(C,R,1,K)/`weight_v`(C,R,K,1)。forward は R 本を順に流して和、bias は最後に1回。
  評価系は `weight_h` の shape[1] から R を自動復元(`run_config.json` の `separable_rank` を優先)。

### 12.3 設計検証(numpy, torch 非依存で数値的に確認)

| 主張 | 検証 | 結果 |
|---|---|---|
| rank-R SVD 合成 == 切り捨て SVD 参照 | ランダム核を両法で再構成し比較 | max err **3e-16**(厳密) |
| 分離 forward(R本の和)== 合成核での密 conv | R=1/2/3 で forward 比較 | max err **~1e-14**(厳密) |
| R=K で密カーネルに厳密一致 | K2(R=K) vs W、forward vs 密(W) | max err **~1e-14**(分離ではなく密そのもの) |
| タップ削減 | 2RK vs K²(K=11) | R=1:5.5x / R=2:2.75x / R=3:1.83x |

→ **rank-R 分離は数学的に正しく実装され、R で密へ連続に近づく**ことを確認。torch スモーク
(`smoke_test.py`)にも rank-2 separable の forward/勾配/FLOPs 比較を追加済み。

### 12.4 推奨学習レシピ(骨シンチ)

```
CUDA_VISIBLE_DEVICES=0 python3 src/train.py --dw-mode separable --separable-rank 2 \
  --batch-size 16 --tag sep_rank2
```

狙い: 機構A rank-1(境界が甘い)との差分を見る。仮説は「rank-1 energy 0.80→rank-2 で ~0.90+ を
回収し、worst10% assd 2倍の境界劣化が baseline へ戻る。同時に 2.75x のタップ削減で実速度の勝ちは
維持」。これが効けば**「rank を上げれば精度-効率を連続制御でき、AGD も全 rank で保つ」**という
骨シンチ唯一の net-positive(速度を得つつ精度を戻す)になる。検証は baseline / sep_rank1 / sep_rank2 を
`compare_models`(worst×境界)+ `benchmark`(実速度・タップ)+ `erf_sigma_table`(kurtosis 保持)で比較。
注: rank-1 の early-stop(22ep)が収束不足でないか best epoch も確認する。

### 12.5 骨シンチ実測結果 — 仮説棄却・multi-seed で中立確定(val n=955)

rank-2(`sep_rank2`)/ 揃えた rank-1(`sep_rank1`, batch16)/ baseline を worst×境界で比較し、
さらに **seed 42/123/456 の multi-seed 対照**で run-to-run ノイズ帯を測った。**結果は仮説の棄却**:

1. **旧「rank-1 が境界を壊す(§5, assd worst10% 2.31=2倍)」は収束不足のアーティファクトだった。**
   揃えて収束させた rank-1(b16)は assd worst10% 1.145 で baseline(1.155)と同等。あの 2.31 は
   batch64・22ep early-stop の1 run 固有(§12.4 の警告が的中)。→ **そもそも「rank で回収すべき境界劣化」は存在しなかった。**
2. **rank-2 は rank-1 に全指標で劣る**: 領域改善は小さく、境界はむしろ悪化側(assd worst10% 1.69)、
   速度は +1%(タップ倍増で separable の +11% 省コストが消える)。→ **rank を上げる意味なし。**
3. **★ multi-seed でノイズ確定**: rank-1 の worst10% Δ(dice +0.002〜0.004 / precision +0.001〜0.007 /
   assd −0.010〜+0.020)は、**baseline を seed 変えただけの揺れ**(precision −0.0016〜+0.0042=幅0.006、
   **assd +0.007〜+0.53**、base_s123 が assd 1.69 に跳ねる)の**中に完全に収まる**。単一 run の
   precision p=8e-10 も assd 改善も **run-to-run ノイズ**。前回 sep_rank2 の「境界劣化 1.69」も
   base_s123 と同値=seed ノイズであって rank-2 の性質ではなかった。

**§12 総括(確定)**: **rank-k 拡張は骨シンチでは不要。** 精度・境界は baseline と中立(multi-seed で
交絡なく確認)。**唯一の堅い差は決定論的な効率= rank-1 separable で +11% 実速度・params35.86M/
FLOPs18.42G(最小)**。主張は「separable rank-1 は骨シンチで精度を落とさず ~11% 速いタダ飯効率化
(局所タスクゆえ精度ゲインは無い)」。方法論的貢献として「**worst 群の p<1e-9 でも単一 run は信用不可、
multi-seed が必須**」を実証(§5 の ERF 正則化ノイズ問題の再演)。→ 機構の精度価値は RF が効く公開
データ(FIVES)でのみ問える(§11)。

## 11. 次

**本筋は RF が効く公開データ(網膜血管 FIVES)への横展開**(memory 道3)。骨シンチで 3 機構
(A/B/C)は検証済み(σ学習・AGD制御・弱点把握・多ガウス効率設計)なので、そのまま持ち込み
「学習σが血管スケールに適応→精度向上」を狙う。機構C は骨シンチでは機構B を超えないと確定した
ので、骨シンチでの追加チューニング(pad 増しで境界破綻抑制など)は任意。**RF が効く FIVES でこそ
多ガウス混合(異なる血管径=異なるスケールの重ね合わせ)が本領を発揮するか**が次の検証点。

## 図(生成コマンド)

層別 ERF の AGD 比較(baseline が滑らかなガウス、pure_spec が尖り+ハローで崩壊、
learned_sigma が中間、を1枚で):
```
python3 src/compare_erf.py --weights $BASE $SPEC $LEARNED \
  --labels baseline pure_spec learned_sigma --part encoder \
  --input-size 512 --n-samples 30 --out-dir erf_results/compare
```
機構ごとの6面 ERF 図(--tag で衝突回避):
```
python3 src/visualize_erf.py --weights $LEARNED --tag learned_sigma --part encoder \
  --input-size 512 --n-samples 50 --out-dir erf_results/learned_sigma
```
