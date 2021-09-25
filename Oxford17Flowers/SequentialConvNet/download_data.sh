# Dataset image
mkdir data
cd data
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz .
tar -xzvf ./17flower.tgz
# Data split
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/17/datasplits.mat
# Segment ground truth
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/17/trimaps.tgz
tar -xzvf ./trimaps.tgz
# Distance matrices 
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/17/distancematrices17gcfeat06.mat
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/17/distancematrices17itfeat08.mat

