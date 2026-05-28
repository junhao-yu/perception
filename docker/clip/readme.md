此目录用于存放 CLIP 源码以在无网络环境下构建 Docker 镜像。

用法：
  在一台有网络的机器上：
    git clone https://github.com/ultralytics/CLIP.git
  然后将整个 CLIP/ 目录拷贝到此目录下：
    docker/clip/setup.py  （这个文件应该存在）

  之后重新构建镜像：
    docker build -t ffs-jetson -f docker/Dockerfile.jetson .

  如果没有 CLIP 源码，构建时会尝试在线安装（需要网络）。
