# [Computational Design and Optimization of Non-Circular Gears (Eurographics 2020)](https://appsrv.cse.cuhk.edu.hk/~haoxu/projects/compute_gear/)
Hao Xu, Tianwen Fu, Peng Song, Mingjun Zhou, Niloy J. Mitra, and Chi-Wing Fu

## About this project
The following image illustrate this project:
![Our result](https://appsrv.cse.cuhk.edu.hk/~haoxu/projects/compute_gear/figures/teaser.png)
We introduce an automatic method to design non-circular gears, which takes two shapes as inputs.
The generated gears are optimized not only to resemble the input shapes (left) but also to transfer motion continuously and smoothly (middle). Further, our results can be 3D-printed and put to work in practice (right). 
See the implementation details in the [project homepage](https://appsrv.cse.cuhk.edu.hk/~haoxu/projects/compute_gear/).

## Installation
This project is writen in python, and is based on the following packages:
- shapely
- numpy
- matplotlib
- scipy
- openmesh
- pyyaml

## How to use the code:
The entry of the code is the "main" function in the file "main_program.py", in which the function takes two shapes as inputs. 
We prepare a bunch of example input shapes in the directory "silhouette". To use any one of them as the input, you just need to type the file name into the function "find_model_by_name".
Once the program is running, a new directory,  *python_dual_gear/debug/yyyy-mm-dd_hh-mm-ss_shapenames*, will be created.
You can find all the intermediate and final results inside that folder.

## How the code is organized
The code is organized 

## The MIT license?

