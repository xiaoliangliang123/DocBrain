import os
import traceback
import torch
from flask import Flask, request, render_template, jsonify, session, redirect, url_for
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.schema import Document
from langchain.llms.base import LLM  # LangChain LLM基类
from pydantic import PrivateAttr  # 关键：Pydantic私有属性声明
from typing import Any, List, Optional
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForVision2Seq  # 本地模型加载库
from login import auth_bp  # 登录蓝图
from flask import send_file, safe_join

# -------------------- 应用初始化与配置 --------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.register_blueprint(auth_bp)

# 核心路径配置（请根据实际修改）
LOCAL_EMBEDDING_MODEL = "D:\\data\\clangchain_ache\\bge-base-zh\\bge-base-zh-v1.5"
Qwen_MODEL_PATH = "D:\\data\\clangchain_ache\\qwen-2b\\Qwen2-VL-2B-Instruct"
chroma_client = chromadb.PersistentClient(path="chroma_vector_db")
os.makedirs('uploads', exist_ok=True)


# -------------------- 本地Qwen模型适配LangChain（使用Pydantic私有属性） --------------------
class LocalQwenLLM(LLM):
    """使用Pydantic私有属性的本地Qwen模型适配器"""
    model_path: str  # Pydantic公开字段（用户需提供的模型路径）
    _tokenizer: Any = PrivateAttr()  # Pydantic私有属性（不参与验证）
    _model: Any = PrivateAttr()  # Pydantic私有属性（不参与验证）

    def __init__(self, model_path: str, **kwargs):
        """初始化时加载模型和分词器（使用私有属性存储）"""
        super().__init__(model_path=model_path, **kwargs)  # 传递Pydantic公开字段

        # 加载分词器和模型（存储到私有属性）
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self._model.eval()  # 推理模式

    @property
    def _llm_type(self) -> str:
        return "local-qwen-vl"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """使用私有属性访问tokenizer和model"""
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.85,
                repetition_penalty=1.2,
                do_sample=True
            )
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()


# -------------------- 核心功能函数（与原逻辑完全一致，无需修改） --------------------
def load_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        documents = []
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text = page.get_text()
            metadata = {"source": file_path, "page": page_num + 1, "filename": os.path.basename(file_path)}
            documents.append(Document(page_content=text, metadata=metadata))
        doc.close()
        return documents
    except Exception as e:
        print(f"加载PDF失败: {e}")
        return []


def split_text(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(documents)


def get_embeddings():
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
        collection.add(
            embeddings=[embeddings.embed_documents([text.page_content])[0]],
            documents=[text.page_content],
            metadatas=[text.metadata],
            ids=[f"{text.metadata['source']}_{text.metadata['page']}"]
        )
    return collection


def get_answer(collection, question):
    llm = LocalQwenLLM(model_path=Qwen_MODEL_PATH)
    prompt_template = """使用以下上下文内容回答末尾的问题。如果你不知道答案，就说不知道，不要编造答案。

{context}

问题: {question}
答案:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

    results = collection.query(
        query_embeddings=get_embeddings().embed_query(question),
        n_results=20,
        include=["metadatas", "documents"]
    )
    relevant_docs = [Document(page_content=doc, metadata=meta)
                     for doc, meta in zip(results["documents"][0], results["metadatas"][0])]

    answer = chain({"input_documents": relevant_docs, "question": question})["output_text"]
    context_info = [{"document_name": doc.metadata["filename"], "context": doc.page_content}
                    for doc in relevant_docs if doc.metadata.get("filename")]
    return answer, context_info


def get_uploaded_documents():
    try:
        collection = chroma_client.get_collection(name="pdf_vectors")
        metadatas = collection.get(include=["metadatas"])["metadatas"]
        return list({meta["filename"] for meta in metadatas if meta.get("filename")})
    except Exception as e:
        print(f"获取文档列表失败: {e}")
        return []


# -------------------- 路由和钩子（与原代码完全一致） --------------------
@app.route('/', methods=['GET'])
def index():
    if 'user' not in session:
        return redirect(url_for('auth.login_page'))
    return render_template('index.html')


@app.route('/upload_and_save', methods=['POST'])
def upload_and_save():
    if 'user' not in session:
        return jsonify({"error": "请先登录"}), 403
    try:
        file = request.files.get('file1')
        if not file or not file.filename.endswith('.pdf'):
            return jsonify({"error": "请上传有效的PDF文件"}), 400
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        documents = load_pdf(file_path)
        texts = split_text(documents)
        save_to_vector_db(texts)
        return jsonify({"message": "文档已成功保存到向量库"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"上传失败: {str(e)}"}), 500


@app.route('/get_uploaded_documents', methods=['GET'])
def get_uploaded_documents_route():
    if 'user' not in session:
        return jsonify({"error": "请先登录"}), 403
    try:
        return jsonify({"documents": get_uploaded_documents()})
    except Exception as e:
        return jsonify({"error": "获取文档列表失败"}), 500

@app.route('/preview_pdf/<filename>')
def preview_pdf(filename):
    """预览上传的PDF文件（需登录）"""
    if 'user' not in session:
        return redirect(url_for('auth.login_page'))

    try:
        # 安全获取文件路径（防止路径遍历攻击）
        file_path = safe_join('uploads', filename)
        # 检查文件是否存在且是PDF
        if not file_path.endswith('.pdf') or not os.path.exists(file_path):
            return "文件不存在", 404

        # 设置响应头让浏览器预览PDF
        return send_file(file_path, mimetype='application/pdf', as_attachment=False)

    except Exception as e:
        return f"预览失败: {str(e)}", 500

@app.route('/qa', methods=['POST'])
def qa():
    if 'user' not in session:
        return jsonify({"error": "请先登录"}), 403
    try:
        question = request.form.get('question')
        if not question:
            return jsonify({"error": "请输入问题"}), 400
        collection = chroma_client.get_collection(name="pdf_vectors")
        answer, context_info = get_answer(collection, question)
        return jsonify({"answer": answer, "context_info": context_info})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"问答失败: {str(e)}"}), 500


@app.before_request
def protect_core_routes():
    allowed_routes = ['auth.login_page', 'auth.login_api']
    if request.endpoint not in allowed_routes and 'user' not in session:
        return redirect(url_for('auth.login_page'))


if __name__ == "__main__":
    app.run(debug=True)