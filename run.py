import os
import sys
import time
import socket
import threading
import urllib.request
import webbrowser
from pathlib import Path

import psutil
import uvicorn


HOST = "127.0.0.1"
PORT = 8000

BASE_URL = f"http://{HOST}:{PORT}"

PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# 判断端口是否已经释放
# ============================================================

def port_is_free() -> bool:
    with socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM,
    ) as sock:

        sock.settimeout(0.2)

        return (
            sock.connect_ex(
                (HOST, PORT)
            )
            != 0
        )


# ============================================================
# 找出监听指定端口的进程
# ============================================================

def find_port_processes():
    processes = []

    try:
        connections = psutil.net_connections(
            kind="inet"
        )

    except Exception as exc:
        print(
            f"[警告] 无法检查端口进程：{exc}"
        )
        return processes


    for connection in connections:

        if connection.status != psutil.CONN_LISTEN:
            continue

        if not connection.laddr:
            continue

        if connection.laddr.port != PORT:
            continue

        if connection.pid is None:
            continue


        try:
            process = psutil.Process(
                connection.pid
            )

            processes.append(
                process
            )

        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
        ):
            continue


    return processes


# ============================================================
# 判断是不是咱自己的旧 Company RAG
# ============================================================

def is_company_rag_process(
    process: psutil.Process,
) -> bool:

    if process.pid == os.getpid():
        return False


    try:
        cmdline = " ".join(
            process.cmdline()
        ).lower()

    except (
        psutil.NoSuchProcess,
        psutil.AccessDenied,
    ):
        return False


    try:
        cwd = Path(
            process.cwd()
        ).resolve()

    except (
        psutil.NoSuchProcess,
        psutil.AccessDenied,
        FileNotFoundError,
    ):
        cwd = None


    same_project = (
        cwd == PROJECT_ROOT
    )


    looks_like_rag = (
        "run.py" in cmdline
        or "app.main:app" in cmdline
        or "uvicorn" in cmdline
    )


    return (
        same_project
        and looks_like_rag
    )


# ============================================================
# 自动清理旧 Company RAG
# ============================================================

def cleanup_old_server():

    processes = find_port_processes()

    if not processes:
        return


    foreign_processes = []
    rag_processes = []


    for process in processes:

        if is_company_rag_process(
            process
        ):
            rag_processes.append(
                process
            )

        else:
            foreign_processes.append(
                process
            )


    # --------------------------------------------------------
    # 如果 8000 是其他程序占用，不乱杀
    # --------------------------------------------------------

    if foreign_processes:

        print(
            f"[错误] {HOST}:{PORT} "
            "被其他程序占用。"
        )

        for process in foreign_processes:

            try:

                cmdline = " ".join(
                    process.cmdline()
                )

                print(
                    f"PID={process.pid}"
                    f"\n命令={cmdline}"
                )

            except Exception:

                print(
                    f"PID={process.pid}"
                )


        print(
            "\n为了避免误杀，"
            "Company RAG 不会自动关闭它。"
        )

        sys.exit(1)


    # --------------------------------------------------------
    # 发现旧 Company RAG
    # --------------------------------------------------------

    for process in rag_processes:

        print(
            "[启动] 发现旧的 Company RAG："
            f"PID {process.pid}"
        )

        print(
            "[启动] 正在关闭旧进程..."
        )

        try:
            process.terminate()

        except psutil.NoSuchProcess:
            pass


    # --------------------------------------------------------
    # 给旧进程最多 3 秒正常退出
    # --------------------------------------------------------

    gone, alive = psutil.wait_procs(
        rag_processes,
        timeout=3,
    )


    # --------------------------------------------------------
    # 实在不退出才 kill
    # --------------------------------------------------------

    for process in alive:

        print(
            f"[启动] PID {process.pid} "
            "未正常退出，强制关闭..."
        )

        try:
            process.kill()

        except psutil.NoSuchProcess:
            pass


    # --------------------------------------------------------
    # 等端口真正释放
    # --------------------------------------------------------

    deadline = time.time() + 5


    while time.time() < deadline:

        if port_is_free():

            print(
                f"[启动] {HOST}:{PORT} "
                "已释放。"
            )

            return

        time.sleep(0.1)


    print(
        f"[错误] {HOST}:{PORT} "
        "仍未释放。"
    )

    sys.exit(1)


# ============================================================
# 等 FastAPI 真正 ready 后再开浏览器
# ============================================================

def wait_and_open_browser():

    health_url = (
        f"{BASE_URL}/api/health"
    )


    while True:

        try:

            with urllib.request.urlopen(
                health_url,
                timeout=0.5,
            ) as response:

                if response.status == 200:
                    break

        except Exception:
            time.sleep(0.2)


    webbrowser.open(
        BASE_URL
    )


# ============================================================
# main
# ============================================================

def main():

    # 先清旧进程。
    #
    # 注意：
    # 这一步发生在 FastAPI / BGE 启动之前，
    # 所以不会出现“模型都加载完了才发现端口被占用”。
    cleanup_old_server()


    threading.Thread(
        target=wait_and_open_browser,
        daemon=True,
    ).start()


    uvicorn.run(
        "app.main:app",

        host=HOST,
        port=PORT,

        reload=False,
        workers=1,

        log_level="info",
    )


if __name__ == "__main__":
    main()
