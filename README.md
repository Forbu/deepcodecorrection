## Error correction code

Error correction code (https://en.wikipedia.org/wiki/Error_correction_code) is an important part of radio telecommunication network.

The idea of error correction code is to add redundancy in the original transmitted signal in order to be robust to signal corruption.

## Deep learning and transformer

One of the core development of the recent decade is the advent of deep learning and the transformer architecture.
Also one can think of the recent development around "discrete" latent representation of data (like VQ VAE (https://arxiv.org/pdf/1711.00937.pdf) or FSQ (https://arxiv.org/abs/2202.01855).
One can exploit the powerful tools that is deep learning to try to create a proper (neural) error correction code software.

## The idea

![Screenshot 2024-01-27 at 19 48 33](https://github.com/Forbu/deepcodecorrection/assets/11457947/6c20c38d-3d37-4823-836c-d6523c0c0fc3)

The idea is to use a discrete code representation couple with clever transformer (permutation invariant architecture) to try to reproduce the performance of classic error correction code.
