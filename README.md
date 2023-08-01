# MultiQueryAttention

Multi Query Attention (MQA) is an innovative Python package that offers an efficient and flexible implementation of the Multi-Query self-attention mechanism.

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install MultiQueryAttention. You can do this via the following command:
```python
pip install mqa
```

## Usage
Here is a simple example of how to initialize and use the `MultiQueryAttention` class.

```python
import torch
from mqa import MultiQueryAttention

x = torch.rand(4, 10, 512).to('cuda')

attn = MultiQueryAttention(
    d_model=512,
    heads=8,
    attn_impl="triton",
    attn_pdrop=0.1,
    device="cuda"
)

#forward pass
output, attn_weights, past_key_values = attn(x)
```

## Class Documentation

### MultiQueryAttention
The `MultiQueryAttention` class is the core component of this package and provides an implementation of the Multi-Query self-attention mechanism.

#### Initialization
The `MultiQueryAttention` class is initialized with the following parameters:

* **d_model**: Dimensionality of the input.
* **heads**: Number of attention heads.
* **attn_impl**: Attention implementation to use ('triton', 'flash', or 'torch').
* **clip_qkv**: Optional parameter to clip query, key, and value vectors.
* **qk_ln**: Optional Boolean flag to apply layer normalization to the query and key vectors.
* **softmax_scale**: Optional scaling factor for the softmax function.
* **attn_pdrop**: Dropout probability for the attention mechanism.
* **norm_type**: Type of normalization to use (default is 'low_precision_layernorm').
* **fc_type**: Type of fully connected layer to use (default is 'torch').
* **verbose**: Verbosity level (default is 0).
* **device**: Device to run the computations on (default is None, automatically chosen).

#### Forward Method
The forward method of the `MultiQueryAttention` class accepts the following parameters:

* **x**: The input tensor.
* **past_key_value**: Optional tensor containing past key and value vectors.
* **bias**: Optional tensor containing attention bias.
* **attention_mask**: Optional tensor containing the attention mask.
* **causal**: Optional Boolean flag indicating if the attention mechanism is causal (default is True).
* **needs_weights**: Optional Boolean flag indicating if the attention weights are needed (default is False).

The forward method returns the output tensor, the attention weights, and the past key and value vectors.

## Conclusion
The MQA package delivers a flexible and efficient toolset for the implementation of the Multi-Query self-attention mechanism. Designed for ease-of-use and integration, it represents a valuable addition to any PyTorch-based project.
