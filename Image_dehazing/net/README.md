# Image Dehazing

#### 2022.01.07

室外拍攝的影像容易受到自然環境的影響，造成解析度降低，影像除霧就是為了改善這方面的問題。影像除霧早在20年前就已經被提起了，早期除霧方式透過提取不同光譜的特徵來達到影像除霧的效果，近幾年來深度學習大量發展，在影像處理的架構上，卷積神經網絡，對於真實色彩影像的處理上提供了較高的準確度，本專題採用的是以CNN卷積神經網路為基本架構，加入兩個注意機制，為了賦予影像象元不同權重所發展出來的FFA_Net，該機制會賦予不同程度之霧霾其該有之權重，讓主網路架構能關注更有效的訊息。

* 模型架構

![GITHUB](https://github.com/gary5312/project/blob/main/Image_dehazing/net/pic/2.png)

* 成果展示

![GITGUB](https://github.com/gary5312/project/blob/main/Image_dehazing/net/pic/1.png)

* 實景拍攝

![GITHUB](https://github.com/gary5312/project/blob/main/Image_dehazing/net/pic/3-1.png)
