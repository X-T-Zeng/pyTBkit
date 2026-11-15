# 有限群、Python抽象接口与普通表示

Python可以实现MATLAB式的基类继承，也可以通过`abc.ABC/abstractmethod`提供更明确的
抽象契约。现有`TBkit(ABC)`已规范数值模型接口；本轮为有限群新增独立的抽象接口。
官方`src/classes/@group/group.m`实际声明为普通`classdef group`，并非`(Abstract)`；
`Oper < group`是继承关系。新实现按Python对象逻辑区分“一个操作”和“整个群”，
不要求一个Oper同时扮演元素、元素数组和群容器。

| 对象 | 职责 |
| --- | --- |
| `Oper` | 对哈密顿量的一次空间/内部/反幺正操作，保留原接口 |
| `AbstractFiniteGroup` | 有限元素、单位元、乘法、逆元的抽象契约；未实现抽象方法时不可实例化 |
| `FiniteGroup` | 带固定元素标签的有限群；以整数Cayley表验证全部群公理 |
| `MatrixRepresentation` | 一个群上的普通幺正线性表示，验证全部乘法关系 |
| `CharacterTable` | 从用户提供的不可约表示建立并验证完整特征标表，供计算及LaTeX展示 |

## 群的基本操作

```python
from tbkit import dihedral_group, cyclic_group, permutation_group

G = dihedral_group(3)       # D_n阶数为2n；这里D3同构于S3
G.order                    # 6
G.identity                 # (0, 0)
G.multiply((1, 0), (0, 1)) # r s；标签(k,b)表示r**k s**b
G.inverse((1, 0))
G.power((1, 0), -2)
G.element_order((1, 0))     # 3
G.conjugacy_classes()      # 类大小1、2、3
G.center()
G.commutator_subgroup()
rotation = G.generated_subgroup([(1, 0)])
reflection = G.generated_subgroup([(0, 1)])
G.subgroups()              # 完整枚举，超max_subgroups则报错，不返回伪完整列表
G.cosets(reflection, side='left')
G.cosets(reflection, side='right')
G.is_normal(rotation)      # True
G.quotient(rotation)       # 实际阶2商群；非正规子群会报错
```

`FiniteGroup(table, elements=...)`中的table使用从0开始的元素索引；标签可以是可哈希对象。
标签顺序不被偷偷重排，单位元也不必是第0项。每行/列为置换、单位元、结合律都要检查，
不能把有单位元和逆元的非结合Latin loop误当成群。
`FiniteGroup.from_group(custom)`可验证符合抽象接口的用户扩展，拒绝声明与实际乘法矛盾的逆元。
循环群、D_n、置换生成群和从严格矩阵乘法生成群均有构造接口。

此层显式存储O(|G|²)乘法表，结合律检查为O(|G|³)，主要面向小有限群及点群。
置换/矩阵生成有max_order边界，子群数也有显式上限；不是大型置换群的Schreier–Sims替代品。

## 普通表示与精确符号

```python
import sympy as s
from tbkit import MatrixRepresentation, CharacterTable

r = s.Matrix([[-s.Rational(1, 2), -s.sqrt(3)/2],
              [s.sqrt(3)/2, -s.Rational(1, 2)]])
mirror = s.diag(1, -1)
A1 = MatrixRepresentation(G, [s.eye(1) for _ in G], name='A_1')
A2 = MatrixRepresentation(G, [s.Matrix([[(-1)**b]]) for k,b in G], name='A_2')
E  = MatrixRepresentation(G, [r**k * mirror**b for k,b in G], name='E')
table = CharacterTable([A1, A2, E])
print(table.values)         # [[1,1,1], [1,1,-1], [2,-1,0]]
print(table.latex())
E.tensor(E).decompose([A1, A2, E])  # (1,1,1)
```

SymPy矩阵默认保持精确，整数、Rational、根号和复相位不变成浮点数；显式exact模式
拒绝Float原子，不用nsimplify猜测原始模型。参数化矩阵只有在给定假设下能证明全部
关系时才接受，数值化前须用`as_arrays(parameters=...)`绑定参数。
绑定后会重新验证幺正性与乘法关系，拒绝违反原符号假设的取值。
NumPy输入使用显式容差验证，不将数值近似称为精确群论证明。

每个表示满足`D(a*b)=D(a)D(b)`和幺正性。不同表示必须共享同一个带索引的群对象，
不根据“看起来同构”而猜测元素对应。`characters(by_class=True)`返回每个共轭类的特征标；
`inner`、`multiplicity`、`decompose`用于正交性和重数。`CharacterTable`检查共轭类数、
不可约性、行正交性与`sum d**2=|G|`；可以传入精确不可约表示，也可显式使用下节的数值生成器。

`frobenius_schur()`仅用于irrep：+1实型、0复型、-1四元数型。例如C3的复一维表示为0，
Q8的二维表示为-1。这不等于已经实现完整的反幺正Wigner核心表示。

## 自动生成普通复不可约表示

```python
from tbkit import irreducible_representations

G = dihedral_group(4)
irreps = irreducible_representations(G, max_order=64,
                                    tolerance=1e-9, spectral_tolerance=1e-8)
[r.dimension for r in irreps]       # 四个一维表示与一个二维表示
table = CharacterTable.from_group(G)  # 同样生成并验证完整数值表
G.regular_representation().decompose(irreps)  # 每个irrep的重数等于其维数
```

生成器以`L(g)|h>=|g*h>`的正则表示为起点。共轭类平均在每个等型空间上为标量，
将它们的两个Hermitian分量逐步联合分解，得到维数`d_alpha**2`的空间；
再用与左作用对易的右正则作用，在重数空间中取一个`d_alpha`维副本。
两个Hermitian分量都要保留，不能只对称化类平均：C3/A4的复共轭表示必须不同，
Q8的四元数型表示在复数域中仍是二维，不能误分成一维或当作四维实表示。

返回值是`exact=False`的`complex128`表示。全部乘法关系、幺正性、不可约性、行正交、
共轭类数及维数平方和都会检查后才返回。它们是显式容差下的数值证据，不是符号证明。
原有精确SymPy输入与符号投影保留；不会用`nsimplify`猜测浮点结果的“精确根式”。
行名`rho_i`是通用标签，不自动声称Mulliken物理命名；简并空间的基底也不唯一，
不能以两个平台输出的矩阵逐元素不同判定表示不等价。

`max_order`默认64，先检查再分配正则作用的稠密准备数据。这个算法面向小有限群，
不是大型置换群或无限群的算法。`spectral_tolerance`用于范数不超过1的分裂算符，
相邻谱隙在阈值的1到10倍间视为歧义并报错；未分清中心、重数不符合平方数、
表示校验失败时同样抛出`IrrepGenerationError`，不返回半张表。
`tolerance`用于最终表示与字符关系检查。若修改这些阈值，应重新检查已知群和误差。

Numba内核`group_kernels.regular_blocks(table, Q)`直接计算`Q† L(g) Q`，避免构造
全部正则矩阵，可在用户`@njit`管线中调用。它接受已验证的数组；符号化简、
谱分解与群对象流程仍在Python/NumPy/LAPACK层，不宣称整个生成器都被JIT编译。

```bash
python -m pytest python/tests/test_automatic_irreps.py -q
python python/examples/automatic_irreps.py --output work/automatic-irreps
NUMBA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python python/benchmarks/regular_blocks.py
```

四站点例子使用正四面体上的无自旋标量轨道和全部相等的实跃迁，完整Td作用同构于S4。
自动识别对称单态和三维扇区，解析能量为`epsilon+3*t`与三重简并的`epsilon-t`；
用SymPy独立推导哈密顿量和解析基底，并验证数值投影及三重态的复幺正换基不变性。
输出原始复数表示/字符/投影、JSON、LaTeX/Markdown和PNG/PDF/SVG。
示意表仅将显示副本舍入，原始数值数组不被修改，数值表示不因此变成精确表示。

![自动表示与四面体紧束缚模型](examples/assets/automatic-irreps.png)

## 投影、重数与简并子空间

`target.projector(irrep)`给出中央等型投影

`P_alpha = d_alpha/|G| * sum_g conj(chi_alpha(g)) D_target(g)`。

其秩是“重数乘irrep维数”，保留所有等价副本；不自动分开重数空间，也不选择唯一基底。
支持精确SymPy与NumPy/Numba数组实现。`tensor`、`direct_sum`及`restrict(subgroup)`支持
张量积、直和及子群限制。

`on_subspace(Q)`中Q的列是正交归一基，并且必须在群作用下封闭。返回该子空间上的
整块表示；换一组子空间基只做相似变换，不改变特征标。
不能把任意简并本征矢下的逐带对角期望值当成对称本征值；后者需要对称算符在简并块内的
谱分析，或使用整块特征标/投影判定。用于Bloch能带时还必须先构造同一k的小群并提供正确sewing，
这里不自动恢复hr.dat中缺少的位置矩阵或处理k依赖投影表示。

## 整体相位、双群与反幺正操作

现有`finite_group`为`U H U†`闭包忽略整体相位，适合哈密顿量平均。这是作用群的商集，
**不能直接当成Hilbert空间的普通表示群**。例如自旋旋转`U=diag(-i,i)`作用于H的阶数为2，
而`U**2=-I, U**4=I`，保留相位的矩阵群阶数为4：

```python
from tbkit import matrix_group
spinor = matrix_group([s.diag(-s.I, s.I)])
spinor.group.order    # 4
spinor.characters()  # (2,0,-2,0)
```

`matrix_group`不按整体相位合并元素；这可表达已给定矩阵生成元的有限中心扩张，
不等于已经自动构建所有晶体双群/因子系统。反幺正`UK`的U在基底变换下是合同变换，
其普通迹不是不变量。`MatrixRepresentation.from_operators`因此拒绝antiunitary操作，
不把它们冒充普通特征标或普通irrep投影。

抽象群可以取商，不代表给定的Hilbert表示必然能降到商群；只有被商子群中的元素
在该表示中均作用为I，才能直接下降为普通表示。不能用取商操作悄悄删除自旋的-I。

## 有限群CG系数与重数空间

```python
from tbkit import clebsch_gordan, symmetry_adapted_basis

# left/right及irreps必须在同一个带索引群对象上，且是普通幺正表示。
result = clebsch_gordan(left, right, irreps)
result.basis           # 列是耦合ket；行是未耦合基 a*dim(right)+b
result.coefficients    # C[a,b,coupled_column]，即<a,b|coupled_column>
result.multiplicities
result.slices          # 每个irrep对应的连续列区间，含全部副本
result.copies(0)       # 第0个irrep的各个N×d嵌入；缺失扇区返回空tuple

resolved = symmetry_adapted_basis(target, irreps)
h_coupled = resolved.transform(H)    # U† H U；不自动假定H对称或厄米
```

列顺序为**irrep、等价副本、irrep内部指标**。每个副本Q都验证
`D_target(g) Q = Q D_alpha(g)`，比“字符相同”更强：副本的内部基底与所提供irrep严格对齐。
完整U还要验证`U†U=I`及覆盖全部目标维数。irrep列表可以省略目标中不出现的表示，
但不能漏掉出现的扇区、重复等价irrep或传入可约表示。不返回未经验证的部分基底。

矩阵单位`E_ij=d/|G| sum_g conj(D_alpha(g)[i,j]) D_target(g)`在等型空间中作用为
`I_m tensor |i><j|`。选取`E_00`值域的正交种子w_mu，
然后`Q_mu[:,i]=E_i0 w_mu`即可同时得到相互正交的全部副本。
这补充了中央字符投影无法单独提供的重数信息；原`projector`仍保留全部等价副本。

若H和群作用对易，在此列顺序中满足`U† H U = direct_sum_alpha(A_alpha tensor I_d)`。
等价副本可以通过A混合；不能把每个副本当成独立、禁止耦合的物理能级。
种子仍可作U(m)旋转，因此A的具体矩阵不是规范唯一的；能谱、重数和相应不变子空间才是稳定量。
`transform`接受一般算符并原样变换，不静默厄米化，也不保证不对称输入会变成分块形式。

全部表示是常量精确SymPy矩阵时，默认用精确矩阵单位与Hermitian Gram–Schmidt，
不恢复浮点数的“精确值”。带自由参数的表示须先绑定；选择定则和算符H中的符号参数
则可以在得到精确基底后保留。`backend='numpy'/'numba'`显式选择数值路径；
默认数值路径用Numba。数值容差须在0与0.5之间，使投影谱的0/1簇不重叠。
固定群作用下的U应只准备一次，在H参数扫描中复用。`max_dimension`默认64，
张量积在构造大矩阵前检查维数限制。
Numba纯数组`group_kernels.matrix_unit_column`可在用户`@njit`中调用；群对象与符号步骤不被JIT。

这不是SU(2)角动量表的相位约定。既有`quantum.coupled_basis`按Condon–Shortley约定返回
**耦合行、未耦合列**；这里返回ket列，复杂系数下不能忽略共轭方向。
有限群CG允许整个irrep的相位及等价副本规范自由度，不自动施加玻色/费米统计。
反幺正核心表示、投影因子系统、Bloch sewing及Wigner–Eckart/6j重耦合不在本接口范围。

```bash
python -m pytest python/tests/test_finite_group_coupling.py python/tests/test_matlab_coupling_audit.py -q
python python/examples/finite_group_coupling.py --output work/finite-group-coupling
NUMBA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python python/benchmarks/matrix_units.py
```

例子精确构造`E tensor E=A1+A2+E`及交换奇偶，并把隐藏在常数复幺正基底中的两份E
还原为重数哈密顿量。解析能量为`±sqrt(delta²+|t|²)`，每条分支二重简并；
这些双态由D3的二维irrep保护，不是已经建立反幺正Kramers结构。
输出原始数组、JSON、LaTeX/Markdown和PNG/PDF/SVG，图形已实际查看。

![精确有限群CG与等价双态混合](examples/assets/finite-group-coupling.png)

## Numba与展示示例

`group_kernels.associative`和`group_kernels.character_projector`接受纯数组，可在用户
`@njit`中调用。群表与表示验证、符号化简在准备阶段完成；应把算好的投影或数值数组
复用于研究热循环。没有跨模型可变全局缓存，也不宣称所有符号操作会被Numba加速。

```bash
python python/examples/group_theory.py --output work/group-theory
python -m pytest python/tests/test_group_theory.py python/tests/test_matlab_group_audit.py -q
```

示例精确推导无自旋等边三站点紧束缚模型的`A1+E`分解，验证`epsilon+2t`单态与
`epsilon-t`双态，并导出字符表、Cayley表、投影、基底和哈密顿量的LaTeX/Markdown，
以及PNG/PDF/SVG和原始数组。LaTeX源文件导出不等于已编译文档PDF；Jupyter接口也不等于
已运行所有前端。由于绘图PR尚未合并，例子使用独立局部样式，合并后可接入公共样式。

![群论与三站点量子模型](examples/assets/group-theory.png)

最小生成元/群秩及同构/自同构算法、精确代数数域irrep自动生成、规范固定的重数基底/6j重耦合、
反幺正核心表示、投影表示因子系统、
完整空间群小群及无限/Lie群仍待继续实现。MATLAB基类审阅和候选补丁证据见
[audit/GROUP-REVIEW.zh-CN.md](audit/GROUP-REVIEW.zh-CN.md)。
