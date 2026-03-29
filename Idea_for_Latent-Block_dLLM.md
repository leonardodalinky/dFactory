# Idea for Latent-Block dLLM

## Idea 背景

现在的 Diffusion Large Language Model (dLLM) 通常会采用 block-wise 的训练方式，即在训练过程中，将输入的文本序列分割成多个 block，每个 block 包含多个 token。然后，模型会学习如何生成每个 block 的内容。对于每个 block 内的 token 是通过 diffusion process 同时逐步生成的；对于 block 之间，则是通过 auto-regressive 的方式一个个 block 依次生成的。

## Idea

我想提出一种新的训练方式，称为 Latent-Block Diffusion Language Model (Latent-Block dLLM)。

首先，对于 block-wise 的训练方式，我想在每个 block 内部的最后一个 token 的位置引入一个 latent token / variable，这个 latent token 会控制整个 block 的生成。具体的操作方式大概如下：
- 首先，在训练时，本来原来每个 block 的大小为 B (在这个仓库的默认情况下，B=32)，所以原始的情况下整个文本是按照 B 个 token 为一个 block 进行分割的。但是在这种新 idea 的情况下，现在应该先按照 B-1 个 token 为一个 block 进行分割，然后在每个 block 的最后一个 token 的位置插入一个 latent token。
  - 对于这个 latent token，我已经在其原本的 tokenizer 上魔改了，位于 `tokenizers/latent/`，只有在 `tokenizer.json` 里面在 reserved token 上增添了一个新的 <|block_latent|> 这个 token，id 为 156901
- 为了给 latent token 位置上生成的 hidden states (latent) 生成对应的 ground-truth，首先在训练时应该引入额外的一个 latent generator，比如我认为先用 sentence-transformer 中的 `all-MiniLM-L6-v2`，把那 B-1 个 token 对应的文本的 embedding 作为 latent 的 ground-truth。
- 训练时的流程:
  - 先从原始的文本中按照 B-1 个 token 为一个 block 进行分割，然后在每个 block 的最后一个 token 的位置插入一个 latent token，得到的序列长度为 B 个 token（刚好为一个 block）。
  - 记这个序列为 S。我们再定义另一个序列 S'，使得 S' 为原始文本中非 user prompt 部分的 token 全部替换为 <|mask|> 的序列 (token id 为 156895)。这两个序列应该有各自的 position_id 序列。
  - 训练过程就是围绕着 [S', S] 两个序列的 concat 进行（这个想法借鉴于仓库里的 block diffusion 训练方法），然后通过调整 attention mask，来控制(1)只训练 latent token 和(2)给定所有 latent token 的情况下，生成其他 token。每个 sample 同时被两种不同的处理方式，得到两个处理后的 sample （或者说生成两个 mask）：
    - 第一种处理方式：只训练 latent token。这个时候就是调整 S' 序列位置上的 attention mask，使得只有模型只能看到 S' 序列上的 latent, block 和 prompt 部分的 token。相当于现在是只给了用户 prompt，然后要先一次性生成所有的 latent token
    - 第二种处理方式：给定所有 latent token 的情况下，生成其余的 token。这个时候就是调整 [S', S] 上的总体的 attention mask。对于每一个 block，都只允许模型看到这个 block 内部的 token、之前所有 block 的 token 以及当前 block 的 latent token，同时还要注意当前 block 中只有部分 token 是应该被 mask 掉的，不是全 mask 掉，所以 attention mask 要仔细设计。
    - 每个 [S', S] 序列总共要重复两次，两次分别对应上面两种不同的 mask
  - 最终的 loss 是两种 loss 的加权平均，权重应该可以调整
- 在推理时，应该先生成 latent token，然后根据 latent token，再 block-wise 的生成其余的 token

## 可能要修改的文件

- `models/llada2_moe/patch_latent_block.py` - 为了不改变原有代码，latent block 的实现应该通过 patch 的形式，注入到原有的模型中，可见 `models/llada2_moe/modeling_llada2_moe.py`。这里我觉得这种 latent token 的加入以及 mask 的构造应该在模型里面处理，而不在 dataset 里面预先构造，虽然慢一些但是通用性更好。此外，模型里面应该分成 train 的 forward 方法和 inference 的 forward 方法，且这两种方法估计要额外接受 latent generator 作为输入。
- `tasks/train_llada2_bd_latent.py` - 这个文件应该是新的训练脚本，用于训练 Latent-Block dLLM，可能还不存在需要创建，但是总体需要参考 `tasks/train_llada2_bd.py`

**编码需要注意**:
- 尽量不要修改 submodule `VeOmni/` 里面的文件，除非确实需要且用户同意
