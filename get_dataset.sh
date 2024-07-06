!/bin/bash

mkdir data
cd data
curl -o train2014.zip http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip
cd ..
