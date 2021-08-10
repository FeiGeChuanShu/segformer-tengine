## Segformer-tengine  
Segformer semantic segmentation infer by tengine
## 前言：
记录一下Segformer部署在tengine上的折腾过程 - 小飞飞的文章 - 知乎
https://zhuanlan.zhihu.com/p/397735148  
## TODO:
1.融合reshape+transpose+matmul+add成一个op
2.AVX/SSE加速matmul和softmax
## 运行结果：
```  
mkdir build  
cd build  
cmake ..  
make  
./segformer_demo -m ../models/segformer.b0.512x1024.city.tmfile -i ../demo.png  
```  
![image](https://github.com/FeiGeChuanShu/segformer-tengine/blob/main/segformer_result.jpg)  
```
tcco@meet:~/segformer-tengine/build/segformer_demo -m ./segformer.b0.512x1024.city.tmfile -i ./demo.png -r 10  
tengine-lite library version: 1.4-dev  
Repeat [10] min 2774.082 ms, max 3876.586 ms, avg 3147.248 ms
```
## Reference  
https://github.com/NVlabs/SegFormer  
https://github.com/OAID/Tengine  
