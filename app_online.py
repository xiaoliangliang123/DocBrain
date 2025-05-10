import os
import traceback
from flask import Flask, request, render_template, jsonify, session, redirect, url_for
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatTongyi
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.schema import Document
import fitz
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from login import auth_bp  # 导入登录蓝图
from flask import send_file

# -------------------- 应用初始化与配置 --------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # 必须与login.py共享同一个SECRET_KEY

# 注册登录蓝图（所有/auth相关路由生效）
app.register_blueprint(auth_bp)

# 核心环境配置
os.environ["DASHSCOPE_API_KEY"] = "你自己的api key"  # 通义千问API Key
LOCAL_MODEL_PATH = "D:\\data\\clangchain_ache\\bge-base-zh\\bge-base-zh-v1.5"  # 本地嵌入模型路径
chroma_client = chromadb.PersistentClient(path="chroma_vector_db")  # 初始化Chroma向量库

# 创建uploads文件夹（用于存储上传的PDF）
os.makedirs('uploads', exist_ok=True)


# -------------------- 核心功能函数 --------------------
def load_pdf(file_path):
    """加载PDF文件并提取文本"""
    try:
        doc = fitz.open(file_path)
        documents = []
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text = page.get_text()
            metadata = {
                "source": file_path,
                "page": page_num + 1,
                "filename": os.path.basename(file_path)
            }
            documents.append(Document(page_content=text, metadata=metadata))
        doc.close()
        return documents
    except Exception as e:
        print(f"加载PDF失败: {e}")
        return []


def split_text(documents):
    """分割文本为块"""
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(documents)


def get_embeddings():
    """初始化句子嵌入模型"""
    model = SentenceTransformer(LOCAL_MODEL_PATH)

    class EmbedFunction:
        def embed_documents(self, texts):
            return [model.encode(text) for text in texts]

        def embed_query(self, text):
            return model.encode(text)

    return EmbedFunction()


def save_to_vector_db(texts):
    """保存文本块到Chroma向量库"""
    embeddings = get_embeddings()
    collection = chroma_client.get_or_create_collection(name="pdf_vectors")
    for text in texts:
        embedding = embeddings.embed_documents([text.page_content])[0]
        collection.add(
            embeddings=[embedding],
            documents=[text.page_content],
            metadatas=[text.metadata],
            ids=[f"{text.metadata['source']}_{text.metadata['page']}"]
        )
    return collection


def get_answer(collection, question):
    """根据向量库内容生成答案"""
    llm = ChatTongyi()
    prompt_template = """使用以下上下文内容回答末尾的问题。如果你不知道答案，就说不知道，不要编造答案。

{context}

问题: {question}
答案:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

    # 查询相关文档
    results = collection.query(
        query_embeddings=get_embeddings().embed_query(question),
        n_results=5,
        include=["metadatas", "documents"]
    )
    relevant_docs = [Document(page_content=doc, metadata=meta)
                     for doc, meta in zip(results["documents"][0], results["metadatas"][0])]

    # 生成答案和上下文信息
    answer = chain({"input_documents": relevant_docs, "question": question})["output_text"]
    context_info = [{"document_name": doc.metadata["filename"], "context": doc.page_content}
                    for doc in relevant_docs if doc.metadata.get("filename")]

    return answer, context_info


def get_uploaded_documents():
    """获取已上传的文档列表（去重）"""
    try:
        collection = chroma_client.get_collection(name="pdf_vectors")
        metadatas = collection.get(include=["metadatas"])["metadatas"]
        return list({meta["filename"] for meta in metadatas if meta.get("filename")})
    except Exception as e:
        print(f"获取文档列表失败: {e}")
        return []


# -------------------- 核心路由（需登录保护） --------------------
@app.route('/', methods=['GET'])
def index():
    """首页路由（需登录）"""
    if 'user' not in session:
        return redirect(url_for('auth.login_page'))  # 未登录跳转登录页
    return render_template('index.html')  # 确保templates/index.html存在


@app.route('/upload_and_save', methods=['POST'])
def upload_and_save():
    """文件上传并保存到向量库（需登录）"""
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
    """获取已上传文档列表（需登录）"""
    if 'user' not in session:
        return jsonify({"error": "请先登录"}), 403

    try:
        return jsonify({"documents": get_uploaded_documents()})
    except Exception as e:
        return jsonify({"error": "获取文档列表失败"}), 500


# -------------------- 新增：PDF预览功能 --------------------
def safe_join(base_dir, filename):
    """
    安全路径拼接函数（防止路径遍历攻击）
    :param base_dir: 基础目录（如上传文件存储目录）
    :param filename: 用户提供的文件名（从前端传入）
    :return: 安全的绝对路径
    :raises ValueError: 当拼接路径超出基础目录时抛出
    """
    # 将基础目录和文件名转换为绝对路径
    base_abs = os.path.abspath(base_dir)
    file_abs = os.path.abspath(os.path.join(base_abs, filename))

    # 关键安全检查：拼接后的路径必须在基础目录内
    if not file_abs.startswith(base_abs):
        raise ValueError(f"非法文件路径访问: {filename}")

    return file_abs


@app.route('/preview_pdf/<filename>', methods=['GET'])
def preview_pdf(filename):
    """
    预览已上传的PDF文件（需登录）
    :param filename: 要预览的PDF文件名（来自文档列表）
    """
    # 1. 登录状态校验（使用现有session机制）
    if 'user' not in session:
        return redirect(url_for('auth.login_page'))  # 未登录跳转登录页

    try:
        # 2. 安全路径拼接（防止路径遍历攻击）
        pdf_path = safe_join('uploads', filename)

        # 3. 文件有效性校验
        if not os.path.isfile(pdf_path):
            return jsonify({"error": "文件不存在"}), 404
        if not pdf_path.lower().endswith('.pdf'):
            return jsonify({"error": "仅支持预览PDF文件"}), 400

        # 4. 返回PDF文件流（浏览器自动预览）
        return send_file(
            pdf_path,
            mimetype='application/pdf',  # 指定MIME类型为PDF
            as_attachment=False  # 不触发下载，直接预览
        )

    except ValueError as e:
        # 路径安全检查失败时返回403禁止访问
        return jsonify({"error": str(e)}), 403
    except Exception as e:
        # 记录详细错误日志（生产环境建议使用日志系统）
        traceback.print_exc()
        return jsonify({"error": f"预览失败: {str(e)}"}), 500

@app.route('/qa', methods=['POST'])
def qa():
    """智能问答（需登录）"""
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


# -------------------- 全局请求钩子（保护核心路由） --------------------
@app.before_request
def protect_core_routes():
    """全局钩子：未登录用户禁止访问非登录路由"""
    allowed_routes = ['auth.login_page', 'auth.login_api']  # 允许未登录访问的路由
    if request.endpoint not in allowed_routes and 'user' not in session:
        return redirect(url_for('auth.login_page'))


if __name__ == "__main__":
    app.run(debug=True)