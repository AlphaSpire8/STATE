完整结构

```
StateTransitionPerturbationModel(
  (loss_fn): SamplesLoss()
  (pert_encoder): Sequential(
    (0): Linear(in_features=5120, out_features=672, bias=True)
    (1): GELU(approximate='none')
    (2): Dropout(p=0.1, inplace=False)
    (3): Linear(in_features=672, out_features=672, bias=True)
    (4): GELU(approximate='none')
    (5): Dropout(p=0.1, inplace=False)
    (6): Linear(in_features=672, out_features=672, bias=True)
    (7): GELU(approximate='none')
    (8): Dropout(p=0.1, inplace=False)
    (9): Linear(in_features=672, out_features=672, bias=True)
  )
  (basal_encoder): Linear(in_features=18080, out_features=672, bias=True)
  (transformer_backbone): LlamaBidirectionalModel(
    (embed_tokens): Embedding(32000, 672, padding_idx=0)
    (layers): ModuleList(
      (0-3): 4 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=672, out_features=672, bias=False)
          (k_proj): Linear(in_features=672, out_features=672, bias=False)
          (v_proj): Linear(in_features=672, out_features=672, bias=False)
          (o_proj): Linear(in_features=672, out_features=672, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=672, out_features=2688, bias=False)
          (up_proj): Linear(in_features=672, out_features=2688, bias=False)
          (down_proj): Linear(in_features=2688, out_features=672, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((672,), eps=1e-06)
        (post_attention_layernorm): LlamaRMSNorm((672,), eps=1e-06)
      )
    )
    (norm): LlamaRMSNorm((672,), eps=1e-06)
    (rotary_emb): NoRoPE()
  )
  (project_out): Sequential(
    (0): Linear(in_features=672, out_features=672, bias=True)
    (1): GELU(approximate='none')
    (2): Dropout(p=0.1, inplace=False)
    (3): Linear(in_features=672, out_features=672, bias=True)
    (4): GELU(approximate='none')
    (5): Dropout(p=0.1, inplace=False)
    (6): Linear(in_features=672, out_features=672, bias=True)
    (7): GELU(approximate='none')
    (8): Dropout(p=0.1, inplace=False)
    (9): Linear(in_features=672, out_features=18080, bias=True)
  )
  (final_down_then_up): Sequential(
    (0): Linear(in_features=18080, out_features=2260, bias=True)
    (1): GELU(approximate='none')
    (2): Linear(in_features=2260, out_features=18080, bias=True)
  )
  (relu): ReLU()
)
```

简化结构

```
| Name                 | Type           | Params | Mode 
----------------------------------------------------------------
0 | loss_fn              | SamplesLoss    | 0      | train
1 | pert_encoder         | Sequential     | 4.8 M  | train
2 | basal_encoder        | Linear         | 12.2 M | train
3 | transformer_backbone | LlamaModel     | 50.4 M | train
4 | project_out          | Sequential     | 13.5 M | train
5 | final_down_then_up   | Sequential     | 81.7 M | train
6 | relu                 | ReLU           | 0      | train
----------------------------------------------------------------
```

