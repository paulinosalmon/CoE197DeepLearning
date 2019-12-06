# CoE 197-Z Deep Learning Project 2
### PokeGAN V2
##### The goal of this project is to improve existing experiments on GANs that produce new and unique Pokemon characters. The task of creating these new cartoons could inspire Pokemon creators in creating new Pokemon charaters for the franchise. Also, it provides excitement to Pokemon lovers knowing that they can create to Pokemons by merging their favorite Pokemons. 

##### With common problems in existing experiments such as the lack of form of the generated Pokemon and sensibility of the images, the results of the past experiments will not inspire Pokemon creators and will not make Pokemon lovers excited. Because of this, the proponents propose the use of a new dataset to improve the quality of the output, as well as some changes to the code to be able to train the GAN in less epochs but still produce output with similar or better quality. This has been achieved with a *INSERT ARCHI HERE* architecture that was trained over *INSERT TRAINING EPOCHS HERE* that translates to *INSERT TRAINING TIME HERE* in *INSERT COMPUTER HERE*.

## Outputs
- Outputs per 50 epochs can be found under the **newPokemon** directory.

## Datasets
- All images can be found under the **data** directory.

## Dependencies
```
opencv-python
tensorflow
scipy
numpy
```

## Run Instructions
```
python newresize.py
python RGBA2RGB.py
python pokeGAN.py
```
