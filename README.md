原CcGAN (Continuous Conditional GAN) 论文及代码库:

https://arxiv.org/pdf/2011.07466v3

https://github.com/UBCDingXin/improved_CcGAN

---

原CcGAN的输入仅包含连续标签(eg:年龄,角度),
现在考虑增加离散标签(eg:类别),进一步增加生成的可控性

本项目是一个demo, 仅基于原项目的部分代码(UTKFace_64X64以及RC-49_64X64部分)做改动与尝试.
在实现了可行性后,后续再进一步改进网络, 扩展到其他数据集, 以及提高分辨率与生成质量(IS, FID等)

---

### log

- 20250301, 可行性 ✓

  但是除DIversity指标有所提升, 其他指标均有下降, 原因是数据集不满足要求, 下一步考虑使用blender渲染更多图片以训练

