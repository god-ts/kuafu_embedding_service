#!/bin/bash

# 停止容器
sudo docker stop ai_answer

# 删除容器
sudo docker rm ai_answer

# 构建镜像
sudo docker build -t answer_service .

# 运行容器
sudo docker run -d --network=hbt-network --name ai_answer --restart always -p 8787:8787 answer_service


