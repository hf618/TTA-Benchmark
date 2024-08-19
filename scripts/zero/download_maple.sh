#!/bin/bash
pip install gdown

# 检查 checkpoints 目录是否存在
if [ ! -d "checkpoints" ]; then
    mkdir checkpoints
fi

# 如果目录存在，继续执行后续命令
echo "Directory 'checkpoints' is ready."
# 继续执行其他命令

# seed 1
gdown https://drive.google.com/uc?id=1zMFZ04l4FegnFea2fYWzQxWjh_TKHdkt
mv model.pth.tar-2 checkpoints/maple_seed1.pth

# seed 2
gdown https://drive.google.com/uc?id=1hLk4tqv0BRo3fnt6ncElbqCteGPe59DE
mv model.pth.tar-2 checkpoints/maple_seed2.pth

# seed 3
gdown https://drive.google.com/uc?id=1eEMVG8Tsfc9SrzSiawbfycjMNF7g3WpD
mv model.pth.tar-2 checkpoints/maple_seed3.pth

echo "MaPLe weights downloaded. You should now have a 'weights' folder with 'maple_seed1.pth', 'maple_seed2.pth' and 'maple_seed3.pth'"
