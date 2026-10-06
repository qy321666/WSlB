# results_pb —— Paderborn(KAt)对照基准

这一目录放**第二个基准**上的实验产物。它存在的唯一理由:CWRU 每个"类别×负载"
只有一颗轴承、一次录制,所以"类间差异是录制指纹"这个结论**在 CWRU 上无法证伪**。
Paderborn 每类含多颗彼此独立制造、独立损伤的轴承,能做 CWRU 做不到的对照。

## 数据

- 位置:`data/paderborn/<轴承码>/<轴承码>/<工况>_<轴承码>_<序号>.mat`
- 来源:`https://groups.uni-paderborn.de/kat/BearingDataCenter/<码>.rar`
- 已下载 **22 颗**:健康 K001–K006;人工损伤 KA01/03/04/05/06/07/08/09/16/22;
  真实(寿命试验)损伤 KI01/03/04/05/07/08
- 通道:振动 = `Y` 里 `Name=='vibration_1'`,`Raster='HostService'`,64 kHz,约 256001 点(4 s);
  转速从 `Y` 的 `speed` 通道读
- 类别按**损伤位置**(事实表 PDF 的 `Component` 字段):健康 / 内圈 IR / 外圈 OR。
  KA08 的事实表把位置写成图例里不存在的 `AR`(源数据笔误),故排除。
- ⚠️ 窗长必须用 **4096**(64 ms)。64 kHz 下 1024 点只有 16 ms,装不下一个 BPFO 周期,
  实测此时所有轴承的谱几乎完全相同(余弦≈1.000),任何谱诊断都会失效。
- ⚠️ 已知混淆:本目录首批下载的轴承里 **KA 恰好全为外圈、KI 恰好全为内圈**,
  于是"损伤位置"与"人工/真实成因"完全重合。已补下 KA09/16/22 等仍是外圈;
  服务器上 KA 系列没有内圈损伤,若要彻底解耦需引入 KB23/24/27(未下载)。

## 脚本

| 脚本 | 做什么 |
| --- | --- |
| `pb_utils.py` | 数据读取管线 + 从 PDF 解析损伤位置(结果缓存 `data/paderborn/damage_table.json`) |
| `tools/probe_spectra.py` | **跨基准对照**:CWRU 与 Paderborn 用同一度量算类均值谱的一致性 |
| `tools/probe_paderborn.py a` | 跨轴承类内一致性 + 同轴承跨工况(CWRU 0.957 的对应量) |
| `tools/probe_paderborn.py bc` | 同录制 vs 轴承不相交(配对比较)+ 工况迁移 |
| `tools/probe_paderborn.py d` | 孪生迁移:类别正是孪生会建模的 IR/OR 运动学 |

## 训练协议

- ResNet1D(与 CWRU 主实验同一骨干)、Adam 1e-3、batch 64、**200 epoch**
  (40 epoch 时方差极大,随机窗测试只有 0.2~0.85,不足以下任何结论)
- `balanced_split`:每个种子随机取每类 5 颗训练轴承 + 1 颗留出轴承,三类均衡
- 测试指标一律 macro-F1
