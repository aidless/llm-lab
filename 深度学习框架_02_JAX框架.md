# JAX深度学习框架：函数式编程与编译器优化

## 1 JAX的核心哲学：为什么需要另一种框架

### 1.1 定位：可微编程的可组合数值计算库

JAX（Just After eXecution）并非传统意义上的深度学习框架。其创造者——Google的Peter Hawkins、Roy Frostig等——将JAX定位为"面向可微编程的可组合数值计算库"[1]，而非另一个PyTorch或TensorFlow的替代品。这套设计的核心由三个组件构成：**Autograd**（自动微分引擎）、**XLA**（Accelerated Linear Algebra编译器）和**JIT**（即时编译）。三者并非简单的加法关系，而是通过统一的中间表示——JAXPR——实现了深度耦合。

JAX的核心理念可以用一句话概括：**将Python函数视为可变换的数学对象**。在这个框架中，你并不编写一个"模型"，而是编写一个Python函数，然后通过`grad`、`jit`、`vmap`、`pmap`等一系列函数变换（Function Transformations）来派生其梯度、加速其执行、向量化其逻辑、分布其计算。这种设计将深度学习从"构建—配置—训练"的管线模式提升到了"函数组合—变换—优化"的数学编程范式。

### 1.2 范式差异：函数式 vs 命令式

PyTorch和TensorFlow（尤其是Eager模式）采用命令式范式：张量是可变状态容器，`tensor += 1`原地修改值，`model.parameters()`持有随时间步更新的隐式状态。这种设计降低了入门门槛，但引入了一类在科学计算和编译器优化中难以处理的副作用问题。

JAX的立场截然相反：**张量是不可变的（Immutable Tensors）**，所有函数必须是纯函数（Pure Functions）——即相同的输入永远产生相同的输出，没有隐藏状态。这一约束并非教条主义的设计选择，而是为了满足XLA编译器所需的代数属性：如果两个函数调用之间没有隐式状态依赖，编译器就可以安全地重排、融合甚至消除操作。

对比而言：

- **PyTorch范式**：`x += 1`是合法操作，`x`的内存地址不变但值变化，这要求运行时维护每个张量的引用计数和可变缓冲区。
- **JAX范式**：`x = x + 1`创建一个新张量，旧张量保持不变（可被GC回收）。这使编译器可以确知每个操作的输入输出边界，从而执行激进的融合策略。

这种函数式约束的直接产物是JAX中**没有"模型"这种对象**。在PyTorch中，`nn.Module`持有参数（`self.weight`），在训练循环中逐步更新；在JAX中，参数是一个普通的PyTree（Python树的广义张量容器），由用户显式传递给损失函数，函数的输出（梯度）被用于创建新的参数PyTree。这种"无状态模型"的设计模式通常写作：

```python
# JAX风格：参数作为函数输入
def loss_fn(params, x, y):
    pred = forward_fn(params, x)
    return jnp.mean((pred - y) ** 2)

grads = grad(loss_fn)(params, x, y)
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

### 1.3 Everything-is-a-Function：函数变换的哲学

JAX最核心的设计原则是"万物皆函数"（Everything-is-a-Function）。在这个世界观中，神经网络的前向传播、损失计算、参数更新——一切——都是纯函数的组合。而函数变换（Transformations）则是在这些函数之上建立的元编程层。

函数变换可以嵌套和组合：`jit(grad(vmap(fn)))`首先对`fn`做向量化，然后取其梯度，再对结果进行JIT编译。这种组合的代数封闭性是JAX区别于其他框架的数学优雅性所在——变换的输出仍然是函数，可以继续被变换。这构成了所谓的**可微编程路线**（Differentiable Programming Paradigm）[2]：

```
Python函数 → jit → 编译后的函数
Python函数 → grad → 梯度函数
Python函数 → vmap → 批量化的函数
Python函数 → pmap → 分布式的函数
```

与TensorFlow 1.x的静态计算图不同，JAX不需要用户显式构建图。图是在函数被变换时由Tracer动态追踪Python函数执行过程而隐式构造的。与PyTorch的eager模式不同，这个"图"不是用于调试的中间产物，而是被送入XLA编译器进行深度优化的入口。

### 1.4 设计目标：为研究而生的低层次灵活性

JAX的设计目标明确指向了机器学习研究社区。研究人员需要的是对计算的底层控制——梯度的精确语义、加速器的直接调度、内存布局的手动优化——而不是高层封装牺牲的灵活性。JAX选择了"低层次但可组合"的路线：

- **对梯度的控制**：`grad`默认对第一个参数求导，但通过`argnums`可以指定任意位置参数，甚至通过`custom_vjp`定义自定义反向传播规则以覆盖默认的自动微分行为。
- **对加速器的控制**：通过`pmap`和`shard_map`，用户可以精确控制数据如何在多设备间分割，以及哪些collective communication操作被执行。
- **对编译的控制**：`jit`提供`static_argnums`指定哪些参数是编译时常量；`lax.while_loop`和`lax.cond`提供对控制流编译的底层控制。

JAX不做的一件事是提供"开箱即用的训练循环"。它提供了一个可微编程的编译器栈，而不是一个框架。这需要用户自己组合grad、jit、vmap和优化器——但正因如此，研究人员获得了从底层修改每一环节的自由。

---

## 2 JIT编译机制

### 2.1 完整编译流水线

JAX的JIT编译是一条从Python到目标机器码的精密的管道，包含三个主要阶段：**JAXPR构建 → XLA编译 → 后端代码生成**。

```
@jit def f(x): return x + 1
        │
        ▼
第一阶段：Tracing → JAXPR
  Python函数在Tracer（抽象张量）的跟踪下执行，
  每个操作（如add）被记录为JAXPR等式（Equation），
  构建Acyclic Graph（有向无环计算图）

        ▼
第二阶段：XLA编译 → HLO
  JAXPR被降级为XLA的HLO（High-Level Optimizer IR），
  XLA依次执行：HLO简化（代数化简）→
  DCE（死代码消除）→ Buffer分析 → Layout分配 →
  Fusion（操作融合）→ SPMD分区（如适用）

        ▼
第三阶段：后端代码生成
  GPU路径：HLO → LLVM IR → PTX（NVIDIA GPU指令）
  TPU路径：HLO → TPU HLO → TPU指令
  CPU路径：HLO → LLVM IR → 主机代码
```

**第一阶段——Tracing到JAXPR**是理解JIT的关键。当`@jit`装饰的函数第一次被调用时，JAX的Tracer对象代替了真实的数值张量被传入函数。Python的每个操作（如`+`、`*`、`jnp.dot`）都被JAX重载并在Tracer对象上执行时，同时将操作记录到JAXPR表达式中。这不是符号执行——函数确实被Python解释器执行了——但执行过程中所有数据依赖关系都被JAXPR捕获。这意味着控制流（`if`、`for`、`while`）的执行路径由Tracer的抽象值决定，因此**数据依赖的控制流会被静态展开**，而Python-native的控制流中未被执行的路径不会出现在JAXPR中[3]。

**第二阶段——XLA编译**是JAX性能优势的核心。XLA（Accelerated Linear Algebra）最初由Google为TensorFlow开发[4]，是一个针对线性代数运算的特定领域编译器。XLA对HLO图的优化包括：

- **Operation Fusion**：将多个小操作（如`add→multiply→tanh`）融合成单个GPU kernel，避免中间结果的显式内存读写和kernel launch开销。在深度学习中，这通常是最大的性能增益来源。
- **Buffer Assignment**：分析张量的生命周期，通过buffer reuse和in-place更新减少内存分配。
- **Layout Assignment**：为每个张量选择最优的内存布局（如NHWC vs NCHW），以最大化硬件利用率。

### 2.2 JAXPR中间表示

JAXPR是JAX的函数追踪规范的底层表示。JAXPR实例是一组等式（Equations）的集合，每个等式定义一个或多个输出（变量）与操作（Primitive）的对应关系：

```
JAXPR结构（概念性）：
  {等式1: {primitive: 'add', inputs: [a, b], output: c},
   等式2: {primitive: 'sin', inputs: [c], output: d},
   ...}

可用 jax.make_jaxpr(f)(x) 查看具体的JAXPR表示。
```

JAXPR中的每个原始操作（Primitive）都对应一个在XLA中有精确语义的原子操作——无论是`add`、`dot_general`这样的数学运算，还是`while`、`cond`这样的控制流。JAXPR的构建是**动态追踪**（Tracing）而非**源码解析**（AST Parsing）的结果：Python函数会实际被执行，只是执行过程中所有数值被Tracer替代。这种设计意味着JAXPR可以捕获复杂的、高度动态的计算模式，只要这些模式不依赖于运行时的数值值（而只依赖于形状和类型）。

JAXPR的闭包特性（Closure Property）赋予了JAX强大的组合能力——任何JAX函数的变换（grad、jit、vmap等）都可以作用于另一个已被变换的函数，因为变换的输入和输出始终是JAXPR可表示的。

### 2.3 JIT缓存机制

JAX的JIT编译缓存基于输入张量的**形状（shape）和数据类型（dtype）**构建缓存键（Cache Key）。每次函数被调用时，JAX计算输入参数（更准确地说，是JAXPR等式的输入变量）的抽象签名（Shape+Dtype+弱类型信息），与缓存中的键比对。如果命中，直接调用已编译的二进制代码；否则重新走完整的编译流水线。

缓存键的设计折射出JAX的哲学：函数的计算图结构依赖于张量的形状，但不依赖于张量的具体数值。因此，当输入形状稳定时（如训练中所有batch的形状固定），JIT编译仅发生一次；但如果遇到动态形状（如变长序列的padding mask），则每次新形状都会触发重编译——这是JAX实践中一个常见的性能陷阱。

对于非张量参数（如循环次数`n`、布尔标志`training=True`），JAX提供了`static_argnums`机制。被标记为`static`的参数将参与计算图的构建——改变它们会触发重编译——但它们不参与梯度计算或设备传输：

```python
@partial(jit, static_argnums=(1,))
def repeat_fn(x, n):
    for _ in range(n):
        x = f(x)
    return x
```

这里的`n`被视为计算图结构的一部分。训练中的`training`标志通常也应标记为static，以避免training/inference路径混合。

此外，JIT缓存具有**PyTree结构约束**：JAX中所有数据都是PyTree（Python list/dict/元组的嵌套结构），缓存键包含PyTree的结构信息。这意味着如果传入参数从字典变为元组（即使值相同），也会触发重编译。

### 2.4 caches与重编译优化

对于输入形状频繁变化的应用场景（如强化学习的rollout收集、变长序列的推理），JAX提供了一些缓解策略：

1. **`jax.jit`的`donate_argnums`**：标记哪些参数可以"捐赠"（其内存被覆盖重用），减少重编译时的新内存分配。
2. **`jax.lax.switch`替代动态`if`**：通过编译时的静态分支选择器，将动态条件判断转换为静态计算图的一部分。
3. **AOT Compilation (`jax.jit`的`in_shardings`配合静态shape)**：在已知所有可能形状池的情况下预编译多个kernel，运行时动态选择而非实时编译。

---

## 3 自动微分与向量化

### 3.1 grad：反向模式自动微分

`grad`是JAX最核心的入口函数。给定一个数值函数`f: ℝⁿ → ℝ`，`grad(f)`返回一个新函数，该函数计算`f`在输入点处的梯度。

`grad`的实现基于**Vector-Jacobian Product（VJP）**的反向模式自动微分。当我们追踪`f`的计算过程到JAXPR中时，每一原始的JAXPR操作（如`sin`、`mul`、`dot_general`）都有预定义的VJP规则。这些规则描述了：给定输出关于某个中间变量的梯度，如何反向传播到输入。

本质上，`grad(f)`做了三件事：

1. **Primal Trace**：用JAXPR追踪`f(x)`的前向计算，生成计算图。
2. **等式反向遍历**：从计算图的最终输出（标量损失）开始，沿计算图的逆序应用每个操作的VJP规则。每个操作会向伴随图中添加"伴随节点"（Adjoint Nodes），这些节点计算输入对该操作输出的梯度贡献。
3. **组合反向计算图**：原始计算图与伴随计算图组合成一个复合计算图（JAXPR表达式），该图可以在XLA中整体编译优化。

关键洞察是：**梯度计算本身是一个JAXPR程序**，可以像任何其他JAX程序一样被`jit`编译和`vmap`向量化。`jit(grad(f))`得到的是经过编译器深度优化的梯度函数。

```python
loss = lambda w, x, y: jnp.mean((jnp.dot(w, x) - y) ** 2)
grad_fn = jit(grad(loss))  # 一次性编译整个梯度计算图
grads = grad_fn(w, x, y)
```

### 3.2 Jacobian：前向与反向的优雅组合

Jacobian矩阵的计算展示了JAX变换组合的威力。对于一个函数`f: ℝⁿ → ℝᵐ`，其Jacobian矩阵`J ∈ ℝᵐˣⁿ`可以通过两种方式构造：

- **JVP（Jacobian-Vector Product，前向模式AD）**：`jax.jvp(f, primals, tangents)`计算`J · v`，其中`v`是切线向量。JVP的复杂度是`O(m)`，当`m << n`时更高效。
- **VJP（Vector-Jacobian Product，反向模式AD）**：`jax.vjp(f, x)`返回一个函数，计算`Jᵀ · u`，其中`u`是余切向量。VJP的复杂度是`O(n)`，当`n << m`时更高效。

JAX的`jacfwd`和`jacrev`分别通过重复应用JVP和VJP来计算完整的Jacobian矩阵：

- **`jacfwd(f, x)`**：对输入`x`的每一列（通过one-hot切线向量）应用JVP，逐列构建Jacobian。迭代次数 = 输入维度。
- **`jacrev(f, x)`**：对输出`y`的每一行应用VJP，会逐行构建Jacobian。迭代次数 = 输出维度。

在典型深度学习场景中（如神经网络的参数梯度），输出通常是标量损失，`m=1`，因此`jacrev`（即`grad`）是自然选择。当`f`的输入和输出维度相近时，JAX内部会进行自动的选择比较——这在`jax.jacobian`高阶API中透明地处理。

更值得注意的是，JVP和VJP本身也是可变换的函数：`jit(jvp(f))`、`vmap(jvp(f))`等都合法且高效。

### 3.3 vmap：自动向量化

`vmap`（Vectorizing Map）是JAX中最具生产力的变换之一。给定接受单个样本的函数`f: ℝⁿ → ℝᵐ`，`vmap(f)`返回一个接受batch张量的函数`f_batch: ℝᵇˣⁿ → ℝᵇˣᵐ`，其中batch维度被自动向量化。

`vmap`的实现不依赖于Python层的手动batch loop或重写。它通过**Batching Rule**（批处理规则）机制工作——JAX中的每个原始操作（Primitive）都有一个预定义的批处理规则，描述了该操作在额外batch维度存在时应如何执行。例如，`dot_general`的批处理规则会将batch维度映射到批矩阵乘的batch维度。

当`vmap(f)`作用于输入时，JAX的Tracer会：

1. 在函数的执行过程中，为每个张量额外标记一个batch轴（默认是轴0）。
2. 遇到每个原始操作时，调用该操作的批处理规则，生成支持batch维度的等价格式。
3. 最终生成的JAXPR是整个batch上向量化的计算图。

这意味着`vmap`与`jit`和`grad`天然兼容：

```python
@jit
def single_loss(params, x, y):
    return jnp.mean((model(params, x) - y) ** 2)

# 自动批量损失
batched_loss = vmap(single_loss, in_axes=(None, 0, 0))
# in_axes=(None, 0, 0) 表示params不拆，x和y的轴0是batch维
```

`vmap`的`in_axes`和`out_axes`参数提供了额外的灵活性：可以指定哪些轴的哪些维度被向量化，以及batch维度在输出中的位置。

### 3.4 pmap：跨设备并行

`pmap`（Parallel Map）是`vmap`的多设备版本。`pmap(f)`将`f`复制到多个XLA设备（GPU/TPU）上，并将输入的第一个维度分割到各设备间执行。与其他框架的数据并行不同，`pmap`中跨设备的通信是通过显式的Collective Operations（如`lax.psum`、`lax.pallgather`）在函数体内部完成的。

`pmap`的工作原理基于XLA的**SPMD（Single Program Multiple Data）**分区能力：

1. **设备复制**：函数`f`被复制到`num_devices`个设备上。
2. **数据分割**：输入的第一个维度被均匀切分（shard）到各设备。
3. **SPMD编译**：XLA在编译阶段分析函数体中所有操作的分区方式。对于`psum`等collective操作，XLA自动插入跨设备通信的同步和聚合逻辑。
4. **同步执行**：所有设备同时执行其分片的计算，collective操作点在设备间同步。

```python
# pmap示例：数据并行分布式训练
@pmap
def train_step(params, batch):
    x, y = batch
    grads = grad(loss_fn)(params, x, y)
    grads = lax.psum(grads, axis_name='batch')  # 跨设备梯度平均
    params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
    return params
```

`pmap`的`axis_name`参数必须与函数体内的collective操作匹配，以标识哪个轴是并行分割的。这是JAX分布式计算中一个常见错误：collective操作必须指定正确的轴名，否则行为未定义。

---

## 4 随机数系统的设计

### 4.1 显式随机状态：函数式随机性的实践

NumPy和大多数深度学习框架的随机数生成器（RNG）依赖于全局隐式状态：`np.random.seed(0)`后，每次调用`np.random.randn()`都会静默更新全局状态。这在单线程科学研究中足够好用，但在JAX的函数式世界和并行计算环境中会产生灾难性的问题——非确定性、竞态条件、以及无法复现的结果。

JAX的随机系统设计完全摒弃了全局状态，转而要求用户**显式传递和管理PRNGKey**（伪随机数生成器密钥）[5]。核心API只有少数几个函数：

```python
key = jax.random.PRNGKey(42)      # 从种子创建初始key
key1, key2 = jax.random.split(key)  # 分割key为两个独立子key
val = jax.random.normal(key1, shape=(3,))  # 使用子key生成随机数
```

关键设计原则在于：**PRNGKey本身是不可变的张量（形状为`(2,)`的`uint32`数组）**。生成随机数不会改变key；要生成新的随机数，必须从key中split出新子key。

这种设计迫使开发者显式地编排随机状态的传播，这初看是额外负担，但在实践中带来了确定性、可复现性和并行安全性：

- **确定性**：给定相同的key序列，无论运行多少次或何种环境，随机数序列完全相同。
- **可复现性**：因为随机状态不被隐式修改，断点续跑的随机状态可以精确恢复。
- **并行安全**：每个设备（或每个vmap的批量元素）获得不同的独立子key，消除了传统RNG在多线程/多设备中的竞态和相关性。

### 4.2 Threefry计数器: 无状态随机数生成架构

JAX默认的PRNG实现基于**Threefry**计数器模式[6]，这是一种从块密码（Threefish加密算法）派生出的散列函数。Threefry的设计采用了**Counter-Based PRNG**（基于计数器的伪随机数生成器）架构，其核心思想颠覆了传统LFSR（线性反馈移位寄存器）或Mersenne Twister的范式。

在Threefry方案中：
- 状态（State）是**显式的计数器+密钥对**（即`(2,)`的`uint32`数组，实际上编码了两个32位整数构成的密钥）。
- 随机数的生成不是通过更新状态实现的，而是对计数器值应用Threefry散列函数求得的。
- `split(key)`操作本质上是对密钥空间进行切分：通过调用Threefry散列对输入key进行"放大"和"分裂"，生成多个在统计意义上独立的子密钥。

具体算法层次如下：

1. **Key Split层**：`jax.random.split(key, n)`对输入的`(2,)`密钥施以Threefry散列操作，生成`(n, 2)`的新密钥数组。这些新密钥在统计意义上与原始密钥独立（由Threefry散列的密码学质量保证）。

2. **Uniform采样层**：`jax.random.uniform(key, shape)`使用key对从0开始的整数序列进行Threefry散列，将结果归一化为[0,1)区间的浮点数。

3. **分布变换层**：在uniform采样的基础上，通过逆变换采样（Inverse Transform Sampling）或拒绝采样（Rejection Sampling）等方法，生成normal、bernoulli、categorical等特定分布的随机数。

这种分层架构确保了：

- **可拆分性**：一个key可以被拆分出任意数量的独立子key，使每个计算单元（vmap元素、pmap设备）获得唯一的随机源。
- **长周期无退化**：由于Threefry是散列函数而非迭代函数，不存在周期退化问题——给定不同的计数器值，输出始终统计独立。
- **快速跳跃**：通过改变计数器值而非经历整个状态序列，Threefry可以高效地"跳跃"到序列中的任意位置。

### 4.3 并行环境中的随机性管理

在`pmap`或`vmap`这样的并行计算中，随机数需要满足两个要求：所有并行执行路径获得统计独立的随机数，同时整个计算保持确定性。

`pmap`场景下，JAX推荐的使用模式是利用`jax.random.fold_in`：

```python
key = jax.random.PRNGKey(0)
# 在pmap外部：为每个设备准备初始key
device_keys = jax.random.split(key, num_devices)

@pmap
def train_step(key, batch):
    # 每个设备获得一个独立key
    noise = jax.random.normal(key, shape=(batch_size,))
    # ... 训练逻辑 ...
    return loss
```

`fold_in(key, data)`的操作语义是：将`data`（通常是一个唯一整数标识符，如当前step或设备ID）与`key`进行混合（通过Threefry散列），生成一个新的key。这避免了在`pmap`函数体内调用`split`时可能出现的设备间key同步问题。

对于`vmap`，JAX提供`jax.random.split(key, batch_size)`预先拆分所有子key，然后将子key列表作为函数参数传入——每个批量元素获得不同的key。

JAX随机系统的一个重要工程进步是，即使在GPU上运行，PRNG操作（`normal`、`bernoulli`等）也是符合XLA编译的，因此可以在`jit`编译的代码内部安全使用，且不会破坏可复现性。

---

## 5 Sharding与分布式计算

### 5.1 PartitionSpec与Mesh：分片规范的基础抽象

JAX的分布式计算模型建立在两个核心抽象之上：**Mesh**（物理设备网格）和**PartitionSpec**（命名的轴分片规范）[7]。

**Mesh**定义了物理设备到逻辑计算轴的映射关系。假设有8个GPU，我们可以将它们组织为`(2, 4)`的2D网格：

```python
from jax.sharding import Mesh, PartitionSpec

devices = jax.devices()  # 8 devices
mesh = Mesh(devices, ('data', 'model'))
```

这里`('data', 'model')`是两个逻辑轴的名字。`mesh.shape['data'] = 2, mesh.shape['model'] = 4`，意味着两个设备用于数据并行方向，四个设备用于模型并行方向。

**PartitionSpec**（或简写作`P`）则描述了特定张量在各逻辑轴上的如何分割：

```python
# spec = PartitionSpec('data', None)
# 含义：张量的第一个轴在'data'逻辑轴上分片（split），
#       第二轴不分割（replicate），即所有设备持有完整第二轴
P = PartitionSpec

spec = P('data', None)
# 张量形状 [8, 1024]:
# 在data轴方向上：每个(data, model)位置的设备持有 [4, 1024]
# 在model轴方向上：所有设备持有完整的 [4, 1024]
```

`None`表示该维度被完整复制（replicate），`'data'`表示该维度在对应的逻辑轴上被切分。PartitionSpec的灵活性在于可以描述各种并行模式：

- `P('data', None)`：数据并行——batch维被切分，参数被复制。
- `P(None, 'model')`：模型并行——参数被切分，batch被复制。
- `P('data', 'model')`：混合并行——张量在两个维度上同时切分。

### 5.2 自动分片：jit的in_shardings/out_shardings

JAX 2.0引入的`jax.jit`的`in_shardings`/`out_shardings`参数彻底简化了分布式编程体验[8]。之前，用户必须显式使用`pmap`或手动调用`shard_map`来管理分片；现在，只需在jit装饰器中声明输入和输出的分片规范：

```python
from jax.sharding import PartitionSpec as P

@partial(jax.jit,
         in_shardings=(P('data', None), P('data', None)),
         out_shardings=P(None, None))
def matmul(x, w):
    return x @ w
```

这段代码声明了：
- 输入`x`的第一轴在`'data'`方向上切分
- 输入`w`的第一轴在`'data'`方向上切分
- 输出`w`被复制到所有设备

JAX的编译时自动分区机制（SPMD Partitioner）在此基础上推导出图中所有中间结果的分片策略，并自动插入必要的跨设备通信操作（all-gather、reduce-scatter等）。开发者不再需要手动推理通信模式。

自动分片的核心优势在于**编写的程序是分片无关**的（shard-unaware）——用户用单设备的方式编写函数，通过声明性的分片规范描述数据分布，编译器负责生成分布式执行计划。

### 5.3 pjit：jit + pmap的统一

`pjit`（现已合并为`jax.jit`的`in_shardings`功能）代表了JAX分布式范式的最终形态：**SPMD的Single-Program自动分区**[9]。

在`pjit`出现之前，JAX用户面临一个分裂的选择：
- 需要单设备或数据并行 → 使用`jit` + `pmap`
- 需要模型并行或混合并行 → 需要手动处理`lax.with_sharding_constraint`和跨设备通信

`pjit`统一了这两个世界。它的工作流程如下：

1. 用户提供输入分片规范（`in_shardings`）和输出分片规范（`out_shardings`）。
2. JAX的SPMD分区器分析函数的JAXPR，根据分片规范推导每个中间操作的**最优分区策略**。
3. 编译器在HLO级别插入必要的collective操作（all-gather、reduce-scatter、all-reduce等）。
4. 编译生成一个在多个设备上执行的XLA计算，每个设备执行其分片部分，并在collective点同步。

这意味着用户可以用同一套API指挥数据并行（`P('data', None)`参数 + `P(None)`梯度）、模型并行（`P(None, 'model')`参数）、全分片数据并行FSDP（将参数和优化器状态在data轴上切分）乃至任意混合策略。

在实际的GPT/Transformer训练中，混合并行策略通常写作三维PartitionSpec：

```python
# Megatron-style混合并行：数据 + 张量 + 流水线
mesh = Mesh(devices, ('dp', 'tp', 'pp'))
# 注意力权重：在tp（张量并行）和dp（数据并行）上切分
attn_weight_spec = P('tp', 'dp')  
# FFN权重：仅tp切分
ffn_weight_spec = P(None, 'tp')
```

---

## 6 生态与框架集成

### 6.1 神经网络库：Flax vs Haiku vs Equinox

JAX不提供内置的神经网络层——它只提供变换原语。社区围绕这一核心开发了三个主要的神经网络库，各自体现了不同的设计哲学。

**Flax**（Google）是当前JAX生态中最流行的神经网络库[10]。其核心是`flax.linen.Module`——一种带有参数初始化包装器和可变状态管理机制的类系统。

```python
class MLP(nn.Module):
    hidden: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x
```

Flax的独特设计在于**Variable Dict**管理机制：Module中所有参数、batch normalization的running stats等由框架自动收集到可区分的变量字典（`{'params': ..., 'batch_stats': ...}`）中。这种设计使得可变状态在函数式范式下得到显式管理——batch stats不会自动更新，而是由函数返回并传递：

```python
def train_step(params, batch_stats, x, y):
    def loss_fn(params):
        y_pred, updated_stats = model.apply(
            {'params': params, 'batch_stats': batch_stats},
            x, mutable=['batch_stats'])
        return jnp.mean((y_pred - y)**2), updated_stats
    
    (loss, new_stats), grads = value_and_grad(loss_fn, has_aux=True)(params)
    params = optax.apply_updates(params, grads)
    return params, new_stats
```

这种"所有可变状态都是函数输出"的设计虽然增加了代码量，但在JIT编译和pmap分布式训练中展现出强大的可预测性——没有隐藏的全局状态被修改。

**Haiku**（DeepMind）采取了另一种路线[11]。它不提供类式的Module，而是提供了一个**函数变换**`hk.transform`，将命令式风格的Module代码转化为纯函数：

```python
def net_fn(x):
    mlp = hk.Sequential([
        hk.Linear(256), jax.nn.relu,
        hk.Linear(10),
    ])
    return mlp(x)

net = hk.without_apply_rng(hk.transform(net_fn))
params = net.init(rng, x)
output = net.apply(params, x)
```

`hk.transform`的语义在类型上等价于JAX的函数变换：它将Module构造与参数初始化（`init`/`apply`的分离）转化为纯函数操作。Haiku的优点是更低层次的透明性——`hk.Sequential`模块不会像Flax那样自动创建`Variable Dict`，所有参数管理通过变换后的接口透明暴露。

**Equinox**（Kidger, 2023）代表了最激进的"JAX风格"[12]。它的核心洞察是：**Module可以是一个PyTree**。任何`eqx.Module`的子类自动成为合法的JAX PyTree，意味着它可以被`jit`、`grad`、`vmap`直接操作——不需要`model.apply`这样的额外步骤：

```python
class MLP(eqx.Module):
    hidden: nn.Linear
    output: nn.Linear
    
    def __call__(self, x):
        x = jax.nn.relu(self.hidden(x))
        return self.output(x)

model = MLP(nn.Linear(784, 256), nn.Linear(256, 10))
# model 是一个 PyTree，可以被直接操作
grads = grad(lambda m, x, y: loss(m(x), y))(model, x, y)
```

Equinox的设计消除了Flax和Haiku中`init`/`apply`的二元性——model本身就是参数，它的`__call__`就是前向函数。这种简洁性使其在JAX生态中迅速增长，但也意味着参数的生命周期管理完全由用户负责。

### 6.2 Optax：梯度变换链

Optax是DeepMind开发的JAX优化器库[13]，其设计哲学体现了JAX的函数式组合美学。

传统优化器是一个聚合对象（如`torch.optim.Adam`），持有参数状态（动量缓存等）。Optax将优化器分解为**梯度变换**（GradientTransformation）的组合链：

```python
import optax

# 组合一个自定义优化器
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),    # 梯度裁剪
    optax.add_weight_decay(1e-4),       # 权重衰减
    optax.scale_by_adam(),              # Adam更新规则
    optax.scale_by_learning_rate(1e-3), # 学习率缩放
)
```

每个`GradientTransformation`是一个实现了`(params, opt_state) → (grads, opt_state)`接口的纯函数。`chain`将多个变换以流水线方式组合：第一个变换的输出作为第二个的输入，以此类推。

Optax的设计优势在于：

- **透明组合**：任何梯度变换都可以与任意优化器组合，不需要预定义的"带weight decay的Adam"这样的变体类。
- **用户自定义**：实现一个新的梯度变换只需实现一个纯函数对，不需要继承优化器基类。
- **JIT兼容**：所有optax变换都是JIT可编译的纯函数操作。

### 6.3 科学计算应用

JAX在科学计算领域的应用正快速增长，其核心吸引力在于同一函数变换体系可以同时服务于深度学习需求（`grad`、`jit`）和科学计算需求（可微编程、GPU加速）。

**JAX-MD**是分子动力学（MD）模拟的JAX实现[14]。它将MD模拟的核心组件——势函数、积分器、邻居列表——全部实现为JAX函数，从而可以：

- 自动微分势能函数的参数（用于力场拟合）
- 用`jit`编译MD步进函数达到数十倍的GPU加速
- 用`vmap`同时模拟多个独立体系

```python
@jit
def md_step(position, velocity, params):
    forces = -grad(energy_fn)(position, params)
    velocity += forces * dt
    position += velocity * dt
    return position, velocity
```

**物理信息神经网络（PINNs）**利用JAX的自动微分能力来求解偏微分方程（PDEs）。PINNs的核心想法是用神经网络近似PDE的解，并用`grad`计算方程中的导数项。JAX的`grad`可以高效计算任意阶导数，配合`jit`编译，使得PINNs的训练在GPU上高效执行。

此外，`jax.lax`模块提供了一系列高级原语（`scan`、`while_loop`、`associative_scan`、`top_k`等），这些原语是科学计算算法（ODE求解器、FFT、矩阵分解）的高效基础。JAX的`lax`设计借鉴了传统的函数式语言（如Haskell的`scanl`、ML族的fold）和线性代数库（LAPACK），但将它们统一在可微分的张量计算框架中。

---

## 7 局限性与发展趋势

### 7.1 陡峭学习曲线：函数式范式的代价

JAX的函数式设计虽然带来了编译器优化的巨大优势，但也导致了显著的学习曲线。三个主要挑战体现在：

**调试困难**。在JAX中，`jit`编译的函数无法被Python的`pdb`逐行调试——函数在Tracer的抽象值上执行，被追踪的是JAXPR而非数值。当出现错误时，XLA编译器报告的堆栈信息往往指向编译后的HLO操作，与原始Python代码的结构无直接对应。JAX的Tracing错误报告（TracerError）通常显示的是JAXPR等式的匹配失败，难以定位到用户代码的具体行。

JAX提供的调试工具包括：

- `jax.debug.print`：编译安全的print操作，在JIT代码中可用的调试打印。
- `jax.debug.breakpoint`：在JIT编译代码中插入断点，暂停执行并进入交互式调试状态。
- `jax.make_jaxpr(f)(x)`：查看函数的JAXPR表示，用于理解计算图的结构。

但这些工具的效率仍然远不如PyTorch的eager模式调试体验。**Eager vs JIT的两难**是JAX框架的根本矛盾：eager模式使调试变得容易但牺牲了性能（JAX的eager模式本质上是在纯Python上执行的NumPy兼容操作，不触发XLA编译），JIT模式提供了极致的性能但调试困难且报错信息晦涩。

**控制流约束**。JAX中循环和条件语句的行为高度依赖于编译时的值。`jax.lax.while_loop`和`jax.lax.cond`提供了可编译的控制流，但它们的语义受到严格限制——循环条件必须在编译时可确定，分支中的值类型必须一致。Python原生`if`/`for`在JIT中会被静态展开，导致非直觉的行为。

**PyTree的认知负担**。所有非张量参数的传递都需要理解PyTree的概念。`jax.tree.map`、`jax.tree_util.tree_reduce`等操作虽然强大，但对初学者可能构成认知障碍。

### 7.2 AOT编译与生态演进

**AOT（Ahead-of-Time）编译**是JAX从2.0开始的重要发展方向[15]。`jax.jit`在默认情况下是JIT——首次调用时编译。AOT编译允许提前完成编译，输出可以序列化保存和跨会话复用：

```python
# AOT编译示例
import jax
from jax import jit

@partial(jit, static_argnums=(0,))
def f(n, x):
    for _ in range(n):
        x = jnp.sin(x)
    return x

# 提前编译
compiled_f = f.lower(100, jnp.ones((1024,)))
compiled_f.compile()  # 或者保存编译结果
```

AOT编译对生产部署意义重大：模型可以在部署前的构建阶段完成编译，避免推理时的编译开销。

**Keras 3的多后端支持**为JAX生态带来了新的发展维度。Keras 3支持JAX作为后端（与TensorFlow、PyTorch并列），这意味着使用Keras API编写的代码可以在不同的后端间切换。对于希望使用高层API但同时利用JAX编译优势的团队，Keras 3提供了一个过渡方案。

**Flax的动态控制流限制**正在被逐步突破。Flax 0.7+提供了`nn.remat`（重材料化，即gradient checkpointing）和更好的`scan`支持。但相比PyTorch在动态unroll方面的成熟，JAX+Flax在高动态控制流（如递归网络、自适应计算）上的支持仍显不足。

### 7.3 JAX 2.0的Roadmap与Pallas内核

JAX 2.0（2023年底发布）引入了几个标志性改进[16]：

1. **统一分片API**：`jit`的`in_shardings`/`out_shardings`替代了`pjit`和`pmap`的独立API。
2. **更好的错误信息**：改进的Tracer错误报告机制，以及`jax.extend.random`中的新随机API。
3. **Pallas内核语言**：Pallas是JAX生态中自定义CUDA kernel的DSL（领域特定语言）。

**Pallas**代表JAX从"编译器用户"向"编译器作者"的重要转变[17]。在Pallas之前，自定义GPU kernel需要在CUDA或Triton中编写，然后通过`jax.experimental.custom_jvp`等接口接入JAX计算图。Pallas允许在Python中直接定义可在XLA编译的kernel：

```python
import jax.experimental.pallas as pl

def add_kernel(x_ref, y_ref, o_ref):
    program_id = pl.program_id(0)  # 获取当前线程索引
    o_ref[program_id] = x_ref[program_id] + y_ref[program_id]

@partial(jit, out_shardings=None)
def pallas_add(x, y):
    return pl.call(lambda: add_kernel, x, y, grid=(x.size,), 
                   out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype))
```

Pallas使得研究人员可以在JAX生态内部完成从算法到GPU kernel的全链路，无需离开Python环境。这是JAX从"高层编译器用户"走向"硬件编程平台"的关键一步。

### 7.4 总结性评估

JAX代表了深度学习框架设计的第三条道路：既非PyTorch的"可调试性优先"，也非TensorFlow的"生产部署优先"，而是**编译器优化优先**的函数式数值计算平台。它在以下场景中具有独特的优势：

- **需要细粒度控制的研究层级自动微分**（如二阶优化器、学习梯度本身的结构）
- **大规模分布式训练**（GSPMD的自动分区推理比手动replica mapping更高效）
- **科学计算与机器学习的交叉领域**（PINNs、可微模拟、逆问题求解）
- **需要极端性能的场景**（XLA操作融合在计算密集模型上效果显著）

它的主要局限性集中体现在调试体验、动态控制流支持、以及社区成熟度三个维度。随着JAX 2/3路线图的推进（Pallas自定义kernel、更好的debug工具、更统一的分布式API），JAX正从"Google内部工具"演进为通用的可微编程基础设施。

---

## 参考文献

[1] Frostig, R., Johnson, M., & Leary, C. (2018). Compiling machine learning programs via high-level tracing. *Systems for Machine Learning (SysML)*.

[2] Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018). Automatic differentiation in machine learning: a survey. *Journal of Machine Learning Research*, 18(153), 1-43.

[3] Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D., Necula, G., Paszke, A., Vanderplas, J., Wanderman-Milne, S., & Zhang, Q. (2018). JAX: Composable transformations of Python+NumPy programs. *GitHub*.

[4] Sabne, A. (2020). XLA: Compiling machine learning for peak performance. *Google Research*.

[5] Johnson, M. J., Frostig, R., Hawkins, P., Wanderman-Milne, S., & Leary, C. (2020). JAX: A differentiated approach to random number generation. *TensorFlow Blog*.

[6] Salmon, J. K., Moraes, M. A., Dror, R. O., & Shaw, D. E. (2011). Parallel random numbers: as easy as 1, 2, 3. *Proceedings of the 2011 International Conference for High Performance Computing, Networking, Storage and Analysis*.

[7] Xu, Y., Lee, H., Chen, D., Hechtman, B., Huang, Y., Jia, R., Choudhury, J., Sinha, A., Kumar, S., & Ghosh, S. (2021). GSPMD: General and scalable parallelization for ML computation graphs. *arXiv preprint arXiv:2105.04663*.

[8] Hechtman, B. (2022). JAX Distributed arrays and automatic parallelization. *Google Research Blog*.

[9] Patwary, M. M. A., Shoeybi, M., Le, L., Smith, S., Holm, J., Fu, Q., Brown, L., Jain, P., & Catanzaro, B. (2021). Efficient large-scale language model training on GPU clusters using Megatron-LM. *SC21*.

[10] Heek, J., Levskaya, A., Oliver, A., Ritter, M., Rondepierre, B., Steiner, A., & van de Meent, J. W. (2020). Flax: A neural network library for JAX. *GitHub*.

[11] Hennigan, T., Cai, T., Norman, T., & Babuschkin, I. (2020). Haiku: Sonnet for JAX. *GitHub*.

[12] Kidger, P. (2021). Equinox: Neural networks in JAX via callable PyTrees. *GitHub*.

[13] Hessel, M., Modayil, J., van Hasselt, H., Schaul, T., Ostrovski, G., Dabney, W., Horgan, D., Piot, B., Azar, M., & Silver, D. (2018). Optax: gradient transformation library for JAX. *GitHub*.

[14] Schoenholz, S. S., & Cubuk, E. D. (2020). JAX MD: A framework for differentiable physics. *Advances in Neural Information Processing Systems*, 33, 11428-11438.

[15] JAX Team. (2023). JAX 2.0 release notes and roadmap. *Google Research*.

[16] Hawkins, P., & Bradbury, J. (2023). JAX: Current state and future directions. *Google I/O*.

[17] JAX Team. (2024). Pallas: A Python kernel language for JAX. *Google Research Technical Report*.
