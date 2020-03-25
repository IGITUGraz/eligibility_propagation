# Getting started

In order to start ATARI experiments you need to carry out the following steps:

- Download the [Arcade-Learning-Environment](https://github.com/mgbellemare/Arcade-Learning-Environment) (ALE) and compile it
- Compile the custom tensorflow op that represents environment steps by running: `python compile.py <path-to-compiled-ALE>`
- Train a spiking agent with reward-based symmetric e-prop by:
`LD_LIBRARY_PATH=<path-to-compiled-ALE>/lib python main.py --result_dir <path-to-a-directory-for-the-results>`

# Acknowledgements

This code is inspired by [https://github.com/deepmind/scalable-agent]
