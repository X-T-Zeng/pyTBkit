# 群论基类与物理使用边界审阅

唯一MATLAB源：BUAA-TCMP/TBkit @ `48e5217593598c8e1d634cdee3976e1dc6984516`。
Python基线：main `13a5832`；不覆盖未合并的符号/绘图/输运分支。

## 抽象类与架构事实

官方`@group/group.m`第一行为`classdef group`，是可实例化基类；`@Oper/Oper.m`继承它。
Python支持真正的`ABC/abstractmethod`契约，现有数值`TBkit`已使用。新增有限群层将单个
`Oper`与群容器分开；FiniteGroup拒绝非结合的乘法、错误的单位元/逆元或非闭合子集。
表示层验证整个普通乘法关系，不依赖isClosed缓存标志。

## MATLAB左单位元错误：原式与最小候选补丁

基类`times`在左因子标记为单位元时执行：

```
G3 = G2;
G3.U = inv(G2);
```

`inv(G2)`返回group对象而非数值矩阵。因此`e .* g`将逆元对象嵌入U，既不满足e*g=g，
也破坏U应表示矩阵的类型；`power`从E开始相乘，其第一次迭代受到同一错误影响。
以90度旋转矩阵验证原式，U成为嵌套group对象，其中矩阵为逆旋转。

候选补丁仅删除错误赋值行，直接返回G2；已加入`official-fixes.patch`。
原始checkout不修改，prepare_upstream在独立official-fixed副本应用补丁。
`test_matlab_group_audit.py`分别运行原式和补丁：原式预期复现嵌套错误，补丁检验左单位元
及1、2、-1幂。Octave适配只移除MATLAB arguments块，并恢复构造器U=nan的原默认值。
没有声称完整MATLAB类/符号工具箱已经验证。

Oper覆盖了times，不能把基类的这一缺陷直接推广为所有Oper乘法都错误。

## 简并态中的对角期望值不是对称本征值

`Character_gen`调用`EIGEN_Kone`，后者计算

`diag(W† V D W)`。

对H=epsilon*I，取D=diag(1,-1)、V=I，W=I与45度旋转后的W都可作为同一简并本征空间的基。
原函数实际执行分别返回(1,-1)和(0,0)，而对称算符谱始终是(+1,-1)，整块特征标始终为0。
这并不是该期望值公式本身的代数错误；它说明不能未经简并块分析、封闭性和小群检查，
就把逐带输出当成不可约表示标签或对称本征值。本轮未改写完整Character_gen流程。

新Python的on_subspace显式验证Q†Q=I与D(g)Q=Q D_sub(g)，只对整块表示取迹。
以D3双态内换基验证特征标不变，非不变子空间明确拒绝。

## 相位商与反幺正操作

现有Python finite_group针对哈密顿量作用忽略整体U相位，这对对称平均是有意设计，
不能把它写成“原Python特征标计算错误”，因为原函数没有提供特征标接口。
新增层补齐了边界：spinor U=diag(-i,i)的严格矩阵群保留-I，阶数4；作用群阶数2。
把这组U硬塞入普通C2表示会违反乘法关系并报错。

对于T=K，U=I时迹为2；仅把一个基矢相位改为i，就有U'=diag(-1,1)、迹为0。
普通特征标公式不适用于antiunitary元素。新接口拒绝该输入，Wigner核心表示留待单独实现。

## 基础功能与独立验证

- C_n元素阶数n/gcd(k,n)，D3/S3共轭类、已知子群阶数及SymPy置换群独立参照。
- D4已知10个子群、6个正规子群，中心及导出子群均为阶2。
- 左/右陪集覆盖、非正规商群拒绝；有单位元/逆元的非结合Latin loop拒绝。
- D3完整特征标表及行/列正交，E tensor E=A1+A2+E，正则表示重数等于irrep维数。
- 三站点P_A1=ones/3、P_E=I-P_A1，解析正交基精确对角化给出单态与二重态能量。
- Q8二维表示的FS=-1，C3复一维FS=0，以及实型D3表示FS=+1。
- 精确/数值投影、随机幺正换基、Numba nopython管线；不以另一个同源公式的一致性代替解析判据。

## 尚需继续审阅的MATLAB源码

源码检查还发现基类cosetseparation循环内未递增count、会覆盖先前陪集；rdivide有将group对象
赋给U的类似写法，inv的数组分支未显式给输出G赋值。本轮没有验证这些完整MATLAB对象数组
流程，也没有对它们提交未经执行验证的候选补丁。新增Python陪集/逆元按独立群公理实现，
不照搬这些分支；不据本轮测试宣称官方group或Oper全部方法已正确移植。

一般不可约表示自动生成、重数空间分离/CG系数、磁群/核心表示、Bloch投影表示和完整
空间群工作流仍未完成。功能与调用约定详见[GROUPS.zh-CN.md](../GROUPS.zh-CN.md)。
