# voice-conversion

1. 训练 
`python train.py case2`
模型保存在`logdir/case2/train2`中
2. 从中断的模型继续训练
` python train2.py case2 -ckpt model-5`
  如果是自行生成的路径，记得有个train2
  
  
  


##### 环境配置
tensorpacK=0.9.1
tf-plot
tensorflow-plot==0.2.0
tensorflow-gpu=1.8.0
tf-plot
tensorflow-plot==0.2.0
yaml
pyyaml
