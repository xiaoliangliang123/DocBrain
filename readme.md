## 简介
> 详情可查看csdn https://blog.csdn.net/madness1010/article/details/147860829
> 500行代码完成的基于flask+langchain+langgrah+chroma+(bge-base-zh-v1.5+Qwen2-VL-2B-Instruct）双大模型 的开源AI智能知识中枢系统‘DocBrain v1.0’，完全免费开源+支持在线和本地化双模式大模型+支持一键式私有化部署+提升工作与学习效率
## 有没有这样一款这开源的本地化轻量级系统？
>我有《哈利波特》的7本电子书，但是没时间看却想知道哈利波特的所有故事线，然后我花了1分钟将这几本电子书上传到了系统里，
然后我输入：告诉我哈利波特从出生到最终所有的故事线和相关人物，一分钟后，系统将7本书里的时间线故事内容整理后以总结的
形式反馈给了我，而且还附加上了每个故事点出自那本书的哪个章节，并且附加了该章节的相关核心内容。
这个系统是什么，对的，这个系统叫DocBrain

## DocBrain是什么
>DocBrain 是一个代码总计不到500行（实际上300多行，下午到现在花了差不多7个小时开发。。。不要问我为什么这么快，因为前端代码我是直接用AI生成的。。）的轻量级系统AI智能知识中枢系统，基于flask+langchain+langgrah+chroma+bge-base-zh-v1.5+Qwen2-VL-2B-Instruct 开发
兼容使用在线大模型和本地大模型接入形式，支持gpu和cpu切换，并支持一键式私有化部署
可以把所有需求的电子文档都收录进去，在需要对内容进行查询时，可以直接输入问题，系统会结合AI按照人类的思维自动
把收录进去的所有内容都检索一遍，把复杂的结果汇总并总结为易理解的内容返回，同时会将
所有相关联的文档名称和对应章节的上下文内容一并列出，安全高效，极大的提升工作和学习效率

## 开发初衷
>描述下开发DocBrain的初衷，因为今天周末休息，没啥事，就打算看一些关于架构涉及的点子书籍和论文学习一下， 
电子文档差不多几百个，但是想找到我需要的内容的相关内容就得一个个看，而且每找到一个还得整理一下，
然后看完另一个之后汇总整理一下，耗时有费力，举个例子，我想从100个电子书里查找关于大数据挖掘的关键技术
核心案例和相关文章出处，就得翻遍这100个文档，然后整理需要内容，最后整体汇总并且记录关键点都出自哪个文档
的哪个段落，上下文都有什么便于下次回顾，十分不方便，而且需要查询并整理另外的知识点需求还要经理另外一边，
因此我就想有没有一个系统，可以把文档都上传上去，然后我想要什么结果就帮我整理成什么结果，完全基于AI，不需要任何
人工成本，我就翻了一圈，发现要么收费，要么开源但是必须使用在线大模型，可以使用本地模型的组件却很多，很重，都不是我想要的
于是直接自己开搞，差不多一上午的时间搞定，用起来速度不错，cpu会慢一些，在线大模型和gpu本地化快的飞起

## 使用方式
>1.登录
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/22a5caaf26794ea682e4524d1d220f75.png)

>2.无文件状态
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3cb8c1a4d7da47018fd1a20e0f487852.png)

>2.上传文件中
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5363fcb6eb8b45a4b140ff36a9d31dc4.png)
>3.上传后
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9658190d048144a9b81b861d439cdfd2.png)

>4.点击预览
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/abdfe3bfcaf645458b160c0ce7c25024.png)
>5.可直接在浏览器预览
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/abfb43e2bc264f958f3c6e8028fbc69d.png)

>6.提问1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7ff41e780f324658b861503eae3720b8.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d36f99d50aaf48ff93d2dea7d61b7b1f.png)



# DocBrain应用场景
1. **学术研究**：上传百篇论文后，可快速整理特定领域的研究现状、关键技术和发展趋势，并标注出处。
2. **法律工作**：将法律法规和案例文档导入，输入问题后系统自动关联相关条款和类似案例。
3. **企业知识管理**：汇总公司内部文档，员工可快速查询业务流程、产品信息等，提升工作效率。
4. **教育培训**：教师上传教材和参考资料，系统可根据教学需求生成知识点总结和案例分析。
5. **技术开发**：收录技术文档和代码示例，开发人员可快速查找解决方案和最佳实践。
6. **文献综述**：自动整合多本学术著作的核心观点，生成结构化的文献综述。
7. **历史研究**：整理多部历史资料中的事件脉络和人物关系，生成时间线和人物图谱。
8. **医疗研究**：帮助医生快速检索医学文献中的治疗方案和临床案例，辅助决策。
9. **项目管理**：整合项目文档，自动生成项目进度报告、风险分析和资源分配建议。
10. **市场调研**：分析行业报告和竞品资料，生成市场趋势和竞争格局分析。

## DocBrain v1.0 功能点介绍
1. **登录**：默认用户密码登录。
2. **pdf上传**：支持pdf格式上传。
3. **上传后预览**：上传后的文档可以点击预览。
4. **文档保存**：上传后文档会保存到向量数据库中。
5. **问题输入**：输入问题。
6. **整理反馈**：根据问题进行扫描汇总以及整理。
7. **关联章节展示**：问题相关的所有文档和相关章节上下文。

## 核心技术介绍
>最近AI agent 比较火，本文用到的也都是agent 相关技术，因此对相关的技术点进行介绍一下，便于初学者小伙伴更容易入门
### 整体流程图
>文档保存流程
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fba662db79444877bcede3fb9a5a72ef.png)
>问题处理流程
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3603699ac0fe40cebcaa7b22fc674e60.png)

### 流程概况

#### 文档上传流程概述
>#### 1. **用户上传文件**：用户通过系统界面上传 PDF 文件，文件数据进入系统处理流程。  
>#### 2. **系统保存文件至系统文件目录**：系统将上传的文件保存到本地的系统文件目录（如 `uploads` 目录），确保文件存储在本地。  
>#### 3. **系统通过 langchain + langgraph 处理文档**：系统利用 `langchain + langgraph` 框架对保存的文档进行初步处理（如文本提取、分割等）。  
>#### 4. **调用 bge - base - zh - v1.5 模型**：`langchain + langgraph` 调用 `bge - base - zh - v1.5` 模型，准备对文档内容进行向量化处理。  
>#### 5. **模型编码生成向量**：`bge - base - zh - v1.5` 模型对文档内容进行 `encode` 操作，将文本转换为向量形式。  
>#### 6. **向量数据调用 langchain + langgraph**：生成的向量数据再次通过 `langchain + langgraph` 进行处理，为存储做准备。  
>#### 7. **保存向量至向量库 chroma**：最后，`langchain + langgraph` 将处理后的向量数据保存到 `chroma` >向量库中，完成文档的向量化存储。  

#### 问题处理流程概述
>#### 1. **用户提交问题（前端触发）**  
>用户在前端界面输入问题（如查询文档相关内容），点击提交按钮，通过网络请求将问题发送至后端指定>接口。

>#### 2. **后端验证登录状态**  
后端程序检查用户登录状态，若用户未登录，返回提示信息要求先登录；若已登录，允许进入下一步处理。

>#### 3. **提取并验证问题内容**  
从请求数据中提取用户输入的问题文本，若问题为空或不合法，返回提示信息要求输入有效问题。

>#### 4. **获取文档向量库集合**  
从数据库或存储系统中获取已保存的文档向量库（存储着上传文档转换后的向量及元数据）。

>#### 5. **将问题转换为向量（BGE 模型编码）**  
利用 `bge-base-zh-v1.5` 模型对问题文本进行编码，将自然语言问题转换为计算机可处理的向量形式。

>#### 6. **在向量库中检索相关文档**  
使用问题向量在向量库中进行检索，找出与问题最相关的文档片段及对应的元数据（如文档来源、页码）。

>#### 7. **整理检索结果为标准文档对象**  
将检索到的文档片段和元数据整理成统一的文档对象格式，方便后续模型处理。

>#### 8. **加载本地 Qwen 模型**  
初始化并加载本地部署的 Qwen 模型，完成模型和分词器的准备工作，使其处于可推理状态。

>#### 9. **构建问答逻辑链条**  
定义包含上下文和问题的提示模板，将文档内容与模型连接，构建起完整的问答逻辑链条，准备生成回答。

>#### 10. **生成回答并返回给用户**  
将整理好的相关文档和问题输入问答链，模型生成回答内容，再将回答及相关上下文信息整理后返回给用户，完成整个问答流程。
### 核心技术点

>将核心代码进行了标记和讲解，便于小伙伴后续学习和二次开发
#### 离线模型代码核心点

```
` app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.register_blueprint(auth_bp)
```

` 
>使用 Flask 构建 Web 应用，SECRET_KEY 用于会话安全，register_blueprint 整合登录路由（auth_bp），实现模块化管理

```
`class LocalQwenLLM(LLM):
    def __init__(self, model_path: str, **kwargs):
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._model = AutoModelForVision2Seq.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        outputs = self._model.generate(**inputs, max_new_tokens=1024, temperature=0.7, top_p=0.85, repetition_penalty=1.2, do_sample=True)
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()`
```

> 定义 LocalQwenLLM 类继承 LLM，利用 Pydantic 私有属性（_tokenizer、_model）封装本地模型加载逻辑。_call 方法处理输入提示，调用本地 Qwen 模型生成回答，实现自定义 LLM 与 LangChain 的集成

`

```
def load_pdf(file_path):
    doc = fitz.open(file_path)
    documents = []
    for page_num in range(doc.page_count):
        text = page.get_text()
        metadata = {"source": file_path, "page": page_num + 1, "filename": os.path.basename(file_path)}
        documents.append(Document(page_content=text, metadata=metadata))
    doc.close()
    return documents`
```

>使用 PyMuPDF（fitz）加载 PDF 文件，逐页提取文本并封装为 LangChain 的 Document 对象（包含文本内容 page_content 和元数据 metadata），为后续处理提供标准化格式

```
`def split_text(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(documents)`
```

>利用 LangChain 的 CharacterTextSplitter 将长文本按固定长度（chunk_size=1000）分割为小块，避免因文本过长导致后续处理（如向量化、模型输入）出现性能问题

```
`def get_embeddings():
    model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
    class EmbedFunction:
        def embed_documents(self, texts):
            return [model.encode(text) for text in texts]
        def embed_query(self, text):
            return model.encode(text)
    return EmbedFunction()

def save_to_vector_db(texts):
    embeddings = get_embeddings()
    collection = chroma_client.get_or_create_collection(name="pdf_vectors")
    for text in texts:
        collection.add(embeddings=[embeddings.embed_documents([text.page_content])[0]], documents=[text.page_content], metadatas=[text.metadata], ids=[f"{text.metadata['source']}_{text.metadata['page']}"])`
```

>通过 SentenceTransformer 加载本地嵌入模型（如 bge-base-zh-v1.5），将文本转换为向量。使用 Chroma 向量数据库存储文本块、向量及元数据，便于后续检索

```
`def get_answer(collection, question):
    llm = LocalQwenLLM(model_path=Qwen_MODEL_PATH)
    prompt_template = """使用以下上下文内容回答末尾的问题。如果你不知道答案，就说不知道，不要编造答案。{context}\n问题: {question}\n答案:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")
    results = collection.query(query_embeddings=get_embeddings().embed_query(question), n_results=20, include=["metadatas", "documents"])
    relevant_docs = [Document(page_content=doc, metadata=meta) for doc, meta in zip(results["documents"][0], results["metadatas"][0])]
    answer = chain({"input_documents": relevant_docs, "question": question})["output_text"]`
```

>定义提示模板 prompt_template，结合 LLMChain（连接 LLM 和提示）与 StuffDocumentsChain（合并文档内容到提示）。通过向量库检索相关文档（collection.query），输入问答链生成最终答案

#### 在线模型代码核心点

```
`os.environ["DASHSCOPE_API_KEY"] = "你自己的模型 api key"
llm = ChatTongyi()`
```

>设置环境变量 DASHSCOPE_API_KEY 为通义千问 API 密钥，通过 ChatTongyi（LangChain 社区提供的类）调用通义千问云端大模型，实现问答逻辑

```
`def get_answer(collection, question):
    llm = ChatTongyi()  # 调用通义千问 API
    prompt_template = """使用以下上下文内容回答末尾的问题。如果你不知道答案，就说不知道，不要编造答案。{context}\n问题: {question}\n答案:"""
    # 构建问答链（同前）
    results = collection.query(query_embeddings=get_embeddings().embed_query(question), n_results=5, include=["metadatas", "documents"])
```
    # 生成答案（同前）`
>与在线代码的核心区别在于 llm 的来源，此代码通过 ChatTongyi 调用通义千问云端 API，而非本地模型，其余问答链构建和答案生成逻辑相似
``
>

## 部署方式
>### 1. **下载代码**  
>### 2. 安装依赖 

```
执行  pip install flask langchain langchain-community pymupdf chromadb sentence-transformers
如果要gpu运行，还要安装torch pip install torch

```

>### 3. 修改模型位置 代码模型改为自己本地路径，如果是在线大模型模式，请改为自己的api key，目前只支持qwen，后续会逐步增加
>使用离线大模型模式模式

```
`LOCAL_EMBEDDING_MODEL = "D:\\data\\clangchain_ache\\bge-base-zh\\bge-base-zh-v1.5"
Qwen_MODEL_PATH = "D:\\data\\clangchain_ache\\qwen-2b\\Qwen2-VL-2B-Instruct"`
```

>使用在线大模型模式

```
`os.environ["DASHSCOPE_API_KEY"] = "你自己的模型 api key`
```

>### 4. **启动** 如果使用在线大模型，则启动app_online.py,  如果使用离线本地大模型，则启动app_offline.py,  
>### 5. **启动浏览** 可以访问 localhost:5000看见系统登录页，默认用户名 admin,密码 123456  

## 离线资源
>离线资源有两个1是bge-base-zh-v1.5，2是Qwen2-VL-2B-Instruct，
> 大家可以直接在魔塔官网下载，我用的都是最小化的模型，都不大，下载速度很快，后续我会将模型上传到网盘，有需要的可以联系我
#### 

## github 源码地址
#### 源码地址：[https://github.com/xiaoliangliang123/DocBrain](https://github.com/xiaoliangliang123/DocBrain)
## 我的微信号：

> unix-blacker，有问题可以一起讨论，原创不易，希望大家github上电量star~




