#!/usr/bin/env bash
# S2 跨工况设置:CWRU load0(源) -> load3(目标),few-shot 目标域
#
# 用法:
#   bash run_s2.sh                # 默认跑现主方法 fp_dann
#   bash run_s2.sh <method>       # source_only | dann | dt_dann | rand_dann | fp_dann | fp_phys_dann
#   bash run_s2.sh all            # 六个方法依次跑(论文主表)
#   bash run_s2.sh smoke          # 快速冒烟:各 2 epoch(验证代码链路)
#
# 常用覆盖变量:
#   K=5 SEED=0 EPOCHS=60 CLASSES=10 SRC=0 TGT=3 DATA=data/cwru OUT=results
#   bash run_s2.sh all            # 正式实验
#   K=5 EPOCHS=2 bash run_s2.sh smoke   # 冒烟
#
# 方法速查(详见 train.py 顶部注释):
#   fp_dann      **现主方法**:目标域谱包络指纹标定 + 白噪声着色生成式增广
#   fp_phys_dann 其物理消融:指纹 + 目标转速冲击串(实测更差)
#   dt_dann      原主方法:物理孪生合成域(已被证否,保留作对照)
#   rand_dann    A3 消融:无物理增广
#   dann         A2 消融:仅对抗域适应
#   source_only  下界

METHOD="${1:-fp_dann}"
K="${K:-5}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-60}"
CLASSES="${CLASSES:-10}"
SRC="${SRC:-0}"
TGT="${TGT:-3}"
DATA="${DATA:-data/cwru}"
OUT="${OUT:-results}"

ALL_METHODS="source_only dann dt_dann rand_dann fp_dann fp_phys_dann"

run_one() {
  python train.py --method "$1" --classes "$CLASSES" \
    --src_load "$SRC" --tgt_load "$TGT" \
    --k "$K" --seed "$SEED" --epochs "$EPOCHS" \
    --data_root "$DATA" --out_dir "$OUT"
}

case "$METHOD" in
  source_only|dann|dt_dann|rand_dann|fp_dann|fp_phys_dann)
    echo ">>> 运行 $METHOD, S$SRC->S$TGT, classes=$CLASSES, k=$K, seed=$SEED, epochs=$EPOCHS"
    run_one "$METHOD"
    ;;
  all)
    for m in $ALL_METHODS; do
      echo ">>> 运行 $m, S$SRC->S$TGT, classes=$CLASSES, k=$K, seed=$SEED, epochs=$EPOCHS"
      run_one "$m"
    done
    ;;
  smoke)
    # 快速链路验证:小 epoch 数跑通全部方法(逻辑冒烟,不追求指标)
    for m in $ALL_METHODS; do
      echo ">>> [smoke] 运行 $m, S$SRC->S$TGT, epochs=2"
      python train.py --method "$m" --classes "$CLASSES" \
        --src_load "$SRC" --tgt_load "$TGT" \
        --k "$K" --seed "$SEED" --epochs 2 \
        --data_root "$DATA" --out_dir "$OUT/smoke"
    done
    ;;
  *)
    echo "未知方法: $METHOD"
    echo "可用: $ALL_METHODS | all | smoke"
    exit 1
    ;;
esac
