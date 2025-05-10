 # login.py
from flask import Blueprint, request, jsonify, session, redirect, url_for, render_template
from werkzeug.security import check_password_hash

# 创建蓝图（名称'auth'，URL前缀'/auth'）
auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

# 模拟用户数据库（实际用真实数据库）
users = {
    "admin": "123456"  # 替换为真实哈希密码
}


@auth_bp.route('/login', methods=['GET'])
def login_page():
    """登录页面路由"""
    return render_template('login.html')


@auth_bp.route('/api/login', methods=['POST'])
def login_api():
    """处理登录请求API"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"code": 400, "msg": "用户名和密码不能为空"})

    if username not in users:
        return jsonify({"code": 401, "msg": "用户名不存在"})

    stored_password = users[username]
    if stored_password == password :
        session['user'] = username  # 存储用户到全局session
        return jsonify({"code": 200, "msg": "登录成功", "redirect": url_for('index')})  # 跳转到首页路由
    else:
        return jsonify({"code": 401, "msg": "密码错误"})


@auth_bp.route('/logout', methods=['GET'])
def logout():
    """退出登录路由"""
    session.pop('user', None)  # 清除session
    return redirect(url_for('auth.login_page'))  # 跳转到登录页