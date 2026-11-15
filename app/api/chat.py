import json
import threading
from queue import Queue
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


router = APIRouter(
    prefix="/api",
    tags=["chat"],
)


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=10000,
    )


def _frontend_result(result: dict[str, Any]) -> dict[str, Any]:
    # raw_answer 不包含 RAGService 追加的“引用来源”文本；
    # 网页使用 sources 自己渲染引用卡片。
    answer = (
        result.get("raw_answer")
        or result.get("answer")
        or ""
    )
    route = result.get("retrieval_route") or {}

    return {
        "answer": answer,
        "sources": result.get("sources") or [],
        "retrieval_mode": route.get("retrieval_mode"),
        "chunk_mode": result.get("chunk_mode"),
        "retrieved_count": result.get("retrieved_count", 0),
    }


@router.post("/chat")
def chat(payload: ChatRequest, request: Request):
    """保留原来的普通 JSON 接口，便于调试或其他调用方继续使用。"""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        service = request.app.state.rag_service
        lock = request.app.state.rag_lock

        with lock:
            result = service.ask(question=question)

        return _frontend_result(result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc


@router.post("/chat/stream")
def chat_stream(payload: ChatRequest, request: Request):
    """
    给浏览器使用的流式状态接口。

    这里不流式生成答案正文，只把真实处理阶段逐条推给前端；
    最终答案仍由原来的 RAGService 一次性返回，所以不改变回答逻辑。
    """
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    service = request.app.state.rag_service
    lock = request.app.state.rag_lock

    def event_stream():
        events: Queue[dict[str, Any] | None] = Queue()

        def progress(message: str) -> None:
            events.put({
                "type": "status",
                "message": message,
            })

        def worker() -> None:
            try:
                with lock:
                    result = service.ask(
                        question=question,
                        progress_callback=progress,
                    )

                events.put({
                    "type": "result",
                    "data": _frontend_result(result),
                })

            except Exception as exc:
                events.put({
                    "type": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                })

            finally:
                events.put(None)

        threading.Thread(
            target=worker,
            daemon=True,
        ).start()

        while True:
            event = events.get()
            if event is None:
                break

            yield json.dumps(
                event,
                ensure_ascii=False,
            ) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "rag_ready": hasattr(
            request.app.state,
            "rag_service",
        ),
    }
