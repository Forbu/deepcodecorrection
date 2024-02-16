## Error correction code

Error correction code (https://en.wikipedia.org/wiki/Error_correction_code) is an important part of radio telecommunication network.

The idea of error correction code is to add redundancy in the original transmitted signal in order to be robust to signal corruption.

## Deep learning and transformer

One of the core development of the recent decade is the advent of deep learning and the transformer architecture.
Also one can think of the recent development around "discrete" latent representation of data like VQ VAE (https://arxiv.org/pdf/1711.00937.pdf) or FSQ (https://arxiv.org/abs/2202.01855).
One can exploit the powerful tools that is deep learning to try to create a proper (neural) error correction code software.

## The idea



The idea is to use a discrete code representation couple with clever transformer (permutation invariant architecture) to try to reproduce the performance of classic error correction code.

![Screenshot 2024-02-15 at 18 52 37](https://github.com/Forbu/deepcodecorrection/assets/11457947/7d07ebac-a6bb-4ab2-a044-fd1da1375a4e)

The "noise" box represent the environment noise that can corrupt the transmitted data.

## Experiences

The first experiences is simple :

Can you make an model that is just able to retrieve the information without any noise but with a discretization layer ?
The result of the first training : 
The answer is (obviously) yes but it takes some times :

![image](https://github.com/Forbu/deepcodecorrection/assets/11457947/ec3630a7-ad4f-4e90-8eb0-ca3935a222c1)

But if we add (channel) noise after the discretization part it leads to instability (no convergence) :

![image](https://github.com/Forbu/deepcodecorrection/assets/11457947/5d22f5c9-8050-47ff-a6e8-89da503dc5af)


And if instead of a discretization layer we add a normalized one (corresponding to power normalization), we obtain better resulting convergence :

![image](https://github.com/Forbu/deepcodecorrection/assets/11457947/45014458-1919-45b9-b4a2-0b439b9f309d)


We observe that an implicit modulation :

![image](https://github.com/Forbu/deepcodecorrection/assets/11457947/a2ee4d1d-9d18-4f2f-b253-d5668016b7b3)




