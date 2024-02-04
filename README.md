## Error correction code

Error correction code (https://en.wikipedia.org/wiki/Error_correction_code) is an important part of radio telecommunication network.

The idea of error correction code is to add redundancy in the original transmitted signal in order to be robust to signal corruption.

## Deep learning and transformer

One of the core development of the recent decade is the advent of deep learning and the transformer architecture.
Also one can think of the recent development around "discrete" latent representation of data like VQ VAE (https://arxiv.org/pdf/1711.00937.pdf) or FSQ (https://arxiv.org/abs/2202.01855).
One can exploit the powerful tools that is deep learning to try to create a proper (neural) error correction code software.

## The idea

![Screenshot 2024-01-27 at 19 54 20](https://github.com/Forbu/deepcodecorrection/assets/11457947/38d1d215-b941-4e3b-a954-973fa5b4df87)
![Screenshot 2024-01-27 at 19 57 04](https://github.com/Forbu/deepcodecorrection/assets/11457947/ce1fe5db-7cb7-46a7-b7b3-62c4fc89fe41)

The idea is to use a discrete code representation couple with clever transformer (permutation invariant architecture) to try to reproduce the performance of classic error correction code.

The "noise" box represent the environment noise that can corrupt the transmitted data.

## Experiences

The first experiences is simple :

Can you make an model that is just able to retrieve the information without any noise ?
The result of the first training : 
TODO add the image

Then we decide to progressively add some noise to see if the transformer is able to create a proper error correcting code
TODO the experience



