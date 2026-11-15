from contextlib import asynccontextmanager
from pathlib import Path
import threading
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router
from app.services.rag import RAGService


ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
STATIC_DIR = WEB_DIR / "static"

# 浏览器页面生命周期管理。
# 多标签页时，只要还有一个页面连接着，就不会关闭后端。
_browser_lock = threading.Lock()
_browser_connections = 0
_browser_ever_connected = False
_shutdown_timer: threading.Timer | None = None

# 刷新页面也会短暂断开 WebSocket，因此留几秒重连宽限期。
BROWSER_CLOSE_GRACE_SECONDS = 5.0


def _cancel_shutdown_timer() -> None:
    global _shutdown_timer

    if _shutdown_timer is not None:
        _shutdown_timer.cancel()
        _shutdown_timer = None


def _request_app_shutdown(app: FastAPI) -> None:
    global _shutdown_timer

    with _browser_lock:
        _shutdown_timer = None

        # 宽限期内已有页面重新连上，说明只是刷新或短暂断线。
        if _browser_connections > 0:
            return

        if not _browser_ever_connected:
            return

    print("[关闭] 浏览器已关闭，Company RAG 自动退出。")

    request_shutdown = getattr(
        app.state,
        "request_shutdown",
        None,
    )

    if callable(request_shutdown):
        request_shutdown()


def _register_browser() -> None:
    global _browser_connections
    global _browser_ever_connected

    with _browser_lock:
        _cancel_shutdown_timer()
        _browser_connections += 1
        _browser_ever_connected = True


def _unregister_browser(app: FastAPI) -> None:
    global _browser_connections
    global _shutdown_timer

    with _browser_lock:
        _browser_connections = max(
            0,
            _browser_connections - 1,
        )

        if _browser_connections > 0:
            return

        _cancel_shutdown_timer()

        _shutdown_timer = threading.Timer(
            BROWSER_CLOSE_GRACE_SECONDS,
            _request_app_shutdown,
            args=(app,),
        )
        _shutdown_timer.daemon = True
        _shutdown_timer.start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[启动] 初始化 RAGService...")

    service = RAGService()

    app.state.rag_service = service
    app.state.rag_lock = threading.Lock()

    print("[启动] RAGService 已就绪。")

    yield

    _cancel_shutdown_timer()
    print("[关闭] Company RAG 已停止。")


app = FastAPI(
    title="Company RAG",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
)

app.include_router(chat_router)

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.websocket("/api/browser-session")
async def browser_session(websocket: WebSocket):
    """
    浏览器页面存活连接。

    - 页面打开：建立 WebSocket。
    - 页面刷新：旧连接断开，但新连接会在宽限期内重新建立。
    - 最后一个页面真正关闭：5 秒后让 Uvicorn 正常退出并释放 8000 端口。
    """
    await websocket.accept()
    _register_browser()

    try:
        while True:
            # 前端会偶尔发送 ping；主要作用是维持并确认连接。
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _unregister_browser(websocket.app)
