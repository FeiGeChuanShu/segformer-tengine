# segformer-tengine
segformer semantic segmentation infer by tengine
## 前言：
浪费了一个周末的时间，先后被onnx导出的一堆琐碎op搞得头大，  
经过op fuse后，又在tengine里面加了gelu，layernorm等新op后，  
终于能够把模型跑起来了，一看运行时间，em...，还我的周末！  
![image](https://github.com/FeiGeChuanShu/segformer-tengine/blob/main/result.jpg)  

