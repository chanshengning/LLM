# ChatGLM-6B

​        ChatGLM-6B是清华大学知识工程和数据挖掘小组发布的一个类似ChatGPT的开源对话机器人，由于该模型是

经过约1T标识符的中英文训练，且大部分都是中文，因此十分适合国内使用。

​        根据GitHub开源项目公开的信息，ChatGLM-6B完整版本需要13GB显存做推理，但是INT4量化版本只需要

6GB显存即可运行，因此对于个人本地部署来说十分友好。本文将详细记录如何基于GPU和部署使用ChatGLM-

6B。

## 1 硬件需求

|    量化等级    | 最低GPU显存（推理） | 最低GPU显存（高效参数微调） |
| :------------: | :-----------------: | :-------------------------: |
| FP16（无量化） |        13GB         |            14GB             |
|      INT8      |         8GB         |             9GB             |
|      INT4      |         6GB         |             7GB             |



##  2 准备环境

### 1.1 新建环境

```shell
conda activate -n chatgbm python=3.8
```

### 1.2 安装依赖

```shell
pip install -r requirements.txt
```

查看requirements.txt文件中要求的包可以点击[这里](https://github.com/THUDM/ChatGLM-6B/blob/main/requirements.txt)

或一个个安装


```shell
pip  install protobuf
```

```shell
pip  install transformers==4.27.1
```

```shell
pip  install cpm_kernels
```

```shell
pip  install mdtex2html
```

```shell
pip  install sentencepiece
```


```shell
pip  install gradio
```

torch安装：

```shell
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

注：在安装torch的时候，根据cuda版本进行安装，且torch>=1.10，[Previous PyTorch Versions | PyTorch](https://pytorch.org/get-started/previous-versions/)



## 2 加载启动模型文件和模型

启动模型文件：Github:https://github.com/THUDM/ChatGLM-6B*

模型文件：Hugging Face Hub(模型地址):https://huggingface.co/THUDM/chatglm-6b*

### 2.1 加载模型启动文件

```git
git clone https://github.com/THUDM/ChatGLM-6B.git
```

以下是下载完成时的样例：

![image-20230423114424398](D:/chan_save/tppora/image-20230423114424398.png)

###  2.2 加载模型文件

```git
git clone https://huggingface.co/THUDM/chatglm-6b
```

以下是下载完成时的样例：

![image-20230423114502182](D:/chan_save/tppora/image-20230423114502182.png)

在下载模型文件和模型的时候我都是使用的Git，在下载的启动文件中新建一个目录存放chatglm-6b模型（这里我新建了/chatglm-6b/用来存放模型文件）

![image-20230423114753578](D:/chan_save/tppora/image-20230423114753578.png)

注：在模型文件中存在以下pytorch开头的文件基本上都是1G以上，模型文件太大，下载可能很慢，可以先下小文件，之后用[清华源](https://so.csdn.net/so/search?q=清华源&spm=1001.2101.3001.7020)下载大模型（我自己下载没出现很慢的情况）

![image-20230423115551747](D:/chan_save/tppora/image-20230423115551747.png)



##  3 模型量化

默认情况下，模型以FP16精度加载，运行模型大概需要13GB显存，如果GPU显存有限，可以尝试量化方式加载模型，使用方法如下：

```python
# 按需修改，目前只支持 4/8 bit 量化
model = AutoModel.from_pretrained("/ChatGLM-6B/chatglm-6b", trust_remote_code=True).quantize(8).half().cuda()
```

进行 2 至 3 轮对话后，8-bit 量化下 GPU 显存占用约为 10GB，4-bit 量化下仅需 6GB 占用。随着对话轮数的增多，对应消耗显存也随之增长，由于采用了相对位置编码，理论上 ChatGLM-6B 支持无限长的 context-length，但总长度超过 2048（训练长度）后性能会逐渐下降。

模型量化会带来一定的性能损失，经过测试，ChatGLM-6B 在 4-bit 量化下仍然能够进行自然流畅的生成。使用 [GPT-Q](https://arxiv.org/abs/2210.17323) 等量化方案可以进一步压缩量化精度/提升相同量化精度下的模型性能。

量化过程需要在内存中首先加载 FP16 格式的模型，消耗大概 13GB 的内存。如果你的内存不足的话，可以直接加载量化后的模型，INT4 量化后的模型仅需大概 5.2GB 的内存：

```python
# INT8 量化的模型将"chatglm-6b-int4"改为"chatglm-6b-int8"
model = AutoModel.from_pretrained("/ChatGLM-6B/chatglm-6b-int4", trust_remote_code=True).half().cuda()
```

量化模型的参数文件可以从[这里](https://cloud.tsinghua.edu.cn/d/674208019e314311ab5c/)手动下载



## 4 部署

多卡部署

如果你有多张 GPU，但是每张 GPU 的显存大小都不足以容纳完整的模型，那么可以将模型切分在多张GPU上。首先安装 accelerate: 

```she
pip install accelerate
```

然后通过如下方法加载模型：

```pyth
from utils import load_model_on_gpus
model = load_model_on_gpus("/ChatGLM-6B/chatglm-6b", num_gpus=3)
```

即可将模型部署到多张 GPU 上进行推理。你可以将 `num_gpus` 改为你希望使用的 GPU 数。默认是均匀切分的，你也可以传入 `device_map` 参数来自己指定。

代码调用

可以通过如下代码调用ChatGLM-6B模型来生成对话：

~~~python
from utils import load_model_on_gpus
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("/ChatGLM-6B/chatglm-6b", trust_remote_code=True, revision="")
model = load_model_on_gpus("/ChatGLM-6B/chatglm-6b", num_gpus=3)
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
~~~

在基于模型启动文件中cli_demo.py，通过3块GPU调用模型，模型代码参看该文件（试过只使用一块GPU，不行）：

cli_demo.py:

```python
# model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
# 替换成
from utils import load_model_on_gpus
model = load_model_on_gpus("/ChatGLM-6B/chatglm-6b", num_gpus=3)
```

```python
用户：你好

ChatGLM-6B：Hello! How can I assist you today?
```

```python
用户：你可以用中文吗

ChatGLM-6B：当然可以，你好！请问有什么我可以帮助你的吗？
```

```python
用户：你可以写代码吗

ChatGLM-6B：当然可以！我是一个大型语言模型，我可以使用各种编程语言来完成任务。不过需要注意的是，我只是一个程序，我的代码只是一系列规则和指令的组合，我不能直接编写代码，我只能通过解释器或编译器将其他编程语言的代码转换成我能够理解的指令和规则。如果你需要编写代码，建议学习编程语言的基础知识和语法，并使用适当的工具来编写和运行代码。
```

````pyth
用户：请写一份单个隐藏式的神经网络

ChatGLM-6B：好的，以下是一个简单的单隐藏层神经网络的代码实现，使用Python和TensorFlow库：

```python
import tensorflow as tf

# 定义神经网络的输入和输出
input_size = 28
output_size = 1

# 定义神经网络的输入层和隐藏层
input_layer = tf.keras.layers.Input(shape=(input_size,))
hidden_layer = tf.keras.layers.隐藏层(num_layers=2, input_shape=(input_size,))
output_layer = tf.keras.layers.Dense(output_size)

# 定义神经网络的模型
model = tf.keras.models.Sequential([input_layer, hidden_layer, output_layer])

# 定义神经网络的权重和偏置
weights = tf.keras.layers.Dense(input_size, activation='relu')(hidden_layer)
biases = tf.keras.layers.Dense(output_size, activation='relu')(output_layer)

# 定义损失函数和优化器
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(output_layer)
optimizer = tf.keras.optimizers.Adam()

# 训练神经网络
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model.fit(input_layer, hidden_layer, epochs=5)

# 使用模型预测
output = model.predict(input_layer)
```

这个模型包含了三个主要组成部分：输入层、隐藏层和输出层。输入层接收输入数据，隐藏层进行特征提取和前向传播，输出层输出预测结果。在这个实现中，隐藏层的数量为2，使用ReLU激活函数，并且每个隐藏层包含2个神经元。损失函数使用SparseCategoricalCrossentropy，优化器使用Adam优化器。训练过程中，我们使用`input_layer, hidden_layer, output`来训练模型，使用`model.predict(input_layer)`来预测输出。
````

```PYTHON
# 结束会话
用户：stop
```

在会话的过程中，回复都是在几秒的时间，稍微慢点



