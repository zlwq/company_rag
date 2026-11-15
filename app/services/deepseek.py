import json
import os
import time
from typing import Any, Callable

import requests


class DeepSeekClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.timeout = int(os.getenv("DEEPSEEK_TIMEOUT", "60"))
        self.retries = int(os.getenv("DEEPSEEK_RETRIES", "4"))
        self.max_tool_rounds = int(os.getenv("DEEPSEEK_TOOL_ROUNDS", "8"))
        self.session = requests.Session()

        if not self.api_key:
            raise RuntimeError("未配置 DEEPSEEK_API_KEY")

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None

        for attempt in range(self.retries):
            try:
                response = self.session.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )

                if response.status_code in {429, 500, 502, 503, 504}:
                    response.raise_for_status()

                response.raise_for_status()
                return response.json()

            except (
                requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError,
            ) as exc:
                last_error = exc
                if attempt + 1 >= self.retries:
                    raise
                time.sleep(min(2**attempt, 6))

        raise RuntimeError("DeepSeek 请求失败") from last_error

    def route_query(self, question: str) -> dict[str, str]:
        system_prompt = """你是企业知识库问答系统的最前置路由器。
你只负责判断回答用户问题需要哪一级公司资料，不负责回答问题，也不负责选择文件。

只输出 JSON：
{"retrieval_mode": "none|metadata_only|document_content", "reason": "一句简短原因"}

判断规则：
1. none：用户只是闲聊、寒暄、通用知识、通用技术原理、概念解释、行业常识，不需要公司内部资料即可回答。
   即使问题里出现了数据库中也可能存在的设备名，只要用户问的是通用知识，也选 none。
   例如：“什么是蓄压器？”、“蓄压器的主要参数都有哪些？”、“PID 是什么？”、“你好”。
2. metadata_only：用户明确在询问公司资料库中的文件级属性，但不需要阅读正文。
   包括文件名、有哪些文件、有哪些版本、日期、路径、文件类型、数量、是否存在某文件等。
   例如：“公司资料库里关于蓄压器的都有哪些版本文件？只列文件名。”
3. document_content：必须阅读公司文件正文才能回答，包括公司方案里的参数、要求、条款、控制方式、报价、内容摘要、局部事实，以及版本/文件正文比较。
4. 用户问题同时包含通用知识和公司资料内容时，选 document_content。
5. 不要因为出现“蓄压器、试验台、PLC”等技术名词就自动选择公司资料；关键看用户是否在问公司/资料库/方案/文件中的信息。
6. 不要自行检索，不要输出除 JSON 以外的内容。
"""

        data = self._post(
            {
                "model": self.model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
            }
        )
        raw = json.loads(data["choices"][0]["message"]["content"].strip())
        mode = str(raw.get("retrieval_mode") or "document_content").strip()
        if mode not in {"none", "metadata_only", "document_content"}:
            mode = "document_content"
        return {
            "retrieval_mode": mode,
            "reason": str(raw.get("reason") or "").strip(),
        }

    @staticmethod
    def _filter_properties() -> dict[str, Any]:
        text_array = {
            "type": "array",
            "items": {"type": "string"},
        }
        return {
            "project_name": text_array,
            "document_type": text_array,
            "version": text_array,
            "customer_name": text_array,
            "file_name": text_array,
            "file_type": text_array,
            "project_stage": text_array,
            "document_date": text_array,
            "source_created_at": text_array,
            "source_modified_at": text_array,
        }

    @classmethod
    def _document_tools(cls) -> list[dict[str, Any]]:
        soft_fields = [
            "project_name",
            "document_type",
            "version",
            "customer_name",
            "file_name",
            "file_type",
            "project_stage",
            "document_date",
            "keywords",
            "summary",
        ]

        return [
            {
                "type": "function",
                "function": {
                    "name": "soft_search_documents",
                    "description": (
                        "第一阶段模糊查找文档。只根据文档 metadata 属性做模糊筛选。"
                        "每个 criterion 内 fields 和 terms 是 OR；多个 criterion 之间是 AND。"
                        "适合主题简称、同义说法、文件类型语义、用户记不准的名称。"
                        "它不会做精确版本判断，也不会自动修正用户输入。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "criteria": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "fields": {
                                            "type": "array",
                                            "items": {
                                                "type": "string",
                                                "enum": soft_fields,
                                            },
                                        },
                                        "terms": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                    },
                                    "required": ["fields", "terms"],
                                },
                            }
                        },
                        "required": ["criteria"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "hard_filter_documents",
                    "description": (
                        "只对当前候选文档做 metadata 机械过滤。filters 做精确等值过滤：字段内多个值是 OR，不同字段之间是 AND；"
                        "required_fields 只要求指定 metadata 字段必须存在且非空。"
                        "只执行你给出的条件，不猜、不修正、不降级。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filters": {
                                "type": "object",
                                "properties": cls._filter_properties(),
                            },
                            "required_fields": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": list(cls._filter_properties().keys()),
                                },
                                "description": "这些字段在最终候选中必须存在且非空。",
                            },
                        },
                        "required": ["filters", "required_fields"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "sort_documents",
                    "description": (
                        "对当前候选文档按可靠时间字段排序，可取最早或最晚的若干文件。"
                        "缺少该时间属性的文件不会被偷偷改用别的时间字段。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "field": {
                                "type": "string",
                                "enum": [
                                    "document_date",
                                    "source_created_at",
                                    "source_modified_at",
                                ],
                            },
                            "order": {
                                "type": "string",
                                "enum": ["asc", "desc"],
                            },
                            "limit": {
                                "type": "integer",
                                "minimum": 1,
                            },
                            "distinct_by": {
                                "type": "string",
                                "enum": ["none", "version", "file_name"],
                                "description": (
                                    "用户说最早两个版本时用 version；最早两个文件时用 none。"
                                    "按 version 选择时，同一版本的重复文件会全部保留。"
                                ),
                            },
                        },
                        "required": ["field", "order", "limit", "distinct_by"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "finalize_documents",
                    "description": (
                        "文档集合已经确定后调用。Soft Search 只是候选召回，不等于文件已经确定。"
                        "chunk_mode=metadata_only 用于只回答文件名、版本、日期、路径、数量等文档属性问题，不读取任何 chunk；"
                        "chunk_mode=all 表示后续把每个选中文件的全部 chunk 原样交给回答模型；"
                        "chunk_mode=relevant 表示只按 detail_queries 检索相关 chunk。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chunk_mode": {
                                "type": "string",
                                "enum": ["metadata_only", "all", "relevant"],
                            },
                            "detail_queries": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 8,
                                "description": (
                                    "仅 relevant 模式使用。把用户问题拆成 1~8 条短而具体的独立检索查询；"
                                    "每条只负责一个信息需求，保留核心对象，并主动加入正文中可能出现的同义词、"
                                    "下位概念或具体字段名。不要直接复制一个很长的用户问题。"
                                ),
                            },
                        },
                        "required": ["chunk_mode", "detail_queries"],
                    },
                },
            },
        ]

    def select_documents(
        self,
        question: str,
        execute_tool: Callable[[str, dict[str, Any]], dict[str, Any]],
        forced_chunk_mode: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        system_prompt = """你是企业知识库的文档检索调度器，不负责回答用户问题。
你的唯一任务是通过工具一步步确定应该使用哪些文件，以及文件确定后如何读取 chunk。

必须遵守以下职责边界：
1. 你负责理解自然语言、判断哪些信息可靠、哪些信息模糊，并决定调用什么工具。
2. Soft Filter 和 Hard Filter 都是被动工具。它们不会替你猜用户意图，不会修正字段，不会 canonical 解析，不会自动降级。
3. 第一步必须调用 soft_search_documents，从文档属性中尽可能完整地找出主题相关文件。
4. 用户明确可靠的精确条件，例如明确版本 V9.4、V3.5，可以在 Soft Filter 返回候选后调用 hard_filter_documents。
5. 如果用户明确表示某个信息“不记得、忘了、可能是、可能写错、大概”，这个信息不可靠，不得用于 Hard Filter。可以完全忽略该不可靠条件，转而使用用户真正可靠的约束。
6. 如果用户要求“最早、最晚、最近、前两个”等，调用 sort_documents。必须选择用户语义真正指向的时间字段；不要把创建时间、文档日期、修改时间互相替代。
7. 同名文件、同版本文件可能有多份且内容不同。不要因为 file_name 或 version 相同就删除、合并或只选一份。工具会用 document_id 区分真实文件。
8. Soft Search 是高召回候选集合，不是最终文件集合。Soft Search 返回结果后，必须检查返回 metadata 是否和用户目标一致；若混有明显不属于目标类别的文件，应继续用 Hard Filter、Sort，或重新调用更具体的 Soft Search 缩小候选，不能把清理工作留给最终回答模型。
9. 如果用户问“有哪些版本的文件 / 哪些版本 / 罗列版本文件”，Soft Search 返回候选后，应调用 Hard Filter，并设置 required_fields=["version"]，机械排除 version=null 的普通相关文件；不要把这些文件一起 finalize。
10. 如果 Soft Filter 或 Hard Filter 返回 0 个文件，不要编造候选。直接调用 finalize_documents，chunk_mode 使用 metadata_only，上层系统会告诉用户资料库没有相应文件。
11. 只有文件集合最终确定后才能调用 finalize_documents。不要提前考虑 chunk。
12. 用户只询问文件名、版本、日期、路径、文件类型、数量等 metadata，不需要正文证据，chunk_mode 必须为 metadata_only，detail_queries 传空数组。
13. 用户要求版本/文件之间的整体区别、全部变化、新增/删除/修改内容，或没有限定具体差异范围时，chunk_mode 必须为 all，确保选中文件的所有 chunk 都提供给回答模型。
14. 用户询问一个或多个明确事实、参数或局部条款时，chunk_mode 使用 relevant，并生成 1~8 个 detail_queries。detail_queries 是给向量检索器使用的检索查询，不是对用户原问题的简单复述，必须遵守：
- 先识别用户问题中彼此独立的信息需求；一个问题包含多个子问题时，分别生成对应的 detail_query。
- 每条 detail_query 只负责一个具体目标，尽量短而具体；不要把多个不同目标塞进一条很长的查询。
- 每条 query 都必须保留核心对象或主题，例如“蓄压器”，避免只写“容积范围”导致检索对象漂移。
- 遇到“参数、物理参数、性能、配置、规格、要求”等抽象上位概念时，必须主动展开为正文中可能实际出现的具体字段、下位概念和同义表达。
- 相近同义词可以放在同一条 query 中，例如“蓄压器 产品容积 容积范围 有效容积”。
- detail_queries 的数量以覆盖用户全部信息需求为准，不为凑数量重复生成近义查询。

例如用户问“蓄压器的物理参数有哪些？”，可拆为：
- 蓄压器 产品容积 容积范围 有效容积
- 蓄压器 工作压力 额定压力 液腔压力
- 蓄压器 充气压力 气腔压力
- 蓄压器 工作介质 氮气 液压油
- 蓄压器 油口 气口 接口规格
- 蓄压器 工作循环频率 响应时间

如果用户同时问“型号是什么、什么类型、为什么使用、物理参数有哪些”，应分别覆盖“型号”“结构/类型”“用途/作用”和各类具体物理参数，而不是把所有词拼成一条长 query。
15. finalize_documents 必须是最后一个工具调用。
16. 如果系统在本轮额外告诉你“固定为 metadata_only”，说明最前置路由器已经确认用户只需要文件 metadata。此时你仍需正常完成文件查找/筛选，但 finalize_documents 必须使用 metadata_only，detail_queries=[]；禁止读取正文。

Soft Filter 的使用方式：
- 一个 criterion 表示一个必须满足的语义条件。
- criterion 内可以列多个字段和多个同义词，只要任意字段匹配任意词就满足该 criterion。
- 多个 criterion 之间全部都要满足。
- 例如要找“蓄压器相关的协议/方案类文件”，可以分别建立“蓄压器主题”和“协议/方案类型”两个 criterion。

例1：
用户：V9.4版蓄压器综合试验台设计方案相比V3.5版，新增、删除或修改了哪些明确的技术要求和控制细节？
正确流程：
- Soft Filter：找所有“蓄压器综合试验台”且属于“设计方案/技术方案/方案”语义的文件。
- Hard Filter：version 精确筛选 V9.4 或 V3.5。
- 不因同名同版本而去重，所有符合条件文件全部保留。
- finalize：chunk_mode=all，因为用户要求完整比较新增、删除、修改。

例2：
用户：帮我看看蓄压器的文件，具体版本是V2还是V3.4我忘了，不记得了，反正就是时间最早的两个版本的文件，然后对比一下他们的区别。
正确流程：
- V2/V3.4 明确被用户声明为不可靠，不使用版本 Hard Filter。
- Soft Filter：根据语境查找蓄压器相关的协议/方案类文件，可主动给出“协议、技术协议、方案、设计方案、计划”等同义词。
- Sort：若用户语义指创建时间，则按 source_created_at 升序取最早 2 个不同 version（distinct_by=version）；若用户明确指文档日期，则使用 document_date。不要自行替换时间含义。
- finalize：chunk_mode=all，因为用户要求整体区别。

例3：
用户：所有文件中关于蓄压器的都有哪些版本的文件？只罗列文件名即可。
正确流程：
- Soft Filter：高召回查找所有蓄压器相关文件。
- 查看 Soft Filter 返回的真实 metadata。因为用户问的是“有版本的文件”，下一步需要版本属性存在。
- Hard Filter：filters={}，required_fields=["version"]，机械排除 version=null 的项目表、确认单或其它普通相关文件。
- finalize：chunk_mode=metadata_only，detail_queries=[]。这个问题只需要文件 metadata，禁止读取正文 chunk。

不要直接回答用户，不要解释你的思考，只通过工具完成检索流程。"""

        if forced_chunk_mode == "metadata_only":
            system_prompt += (
                "\n\n本轮固定证据模式：metadata_only。"
                "你可以正常调用 Soft/Hard/Sort 来确定文件，但 finalize_documents "
                "必须使用 chunk_mode=metadata_only 且 detail_queries=[]。"
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        tools = self._document_tools()
        tool_trace = []

        if progress_callback:
            progress_callback("正在查找相关文件…")

        for round_index in range(self.max_tool_rounds):
            if progress_callback and round_index > 0:
                progress_callback("正在分析候选文件…")
            data = self._post(
                {
                    "model": self.model,
                    "temperature": 0,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                }
            )
            message = data["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                raise RuntimeError(
                    "DeepSeek 文档调度器未调用检索工具，无法继续。"
                    f" 模型输出: {message.get('content') or ''}"
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content"),
                    "tool_calls": tool_calls,
                }
            )

            for tool_call in tool_calls:
                function = tool_call.get("function") or {}
                name = str(function.get("name") or "")
                raw_arguments = function.get("arguments") or "{}"
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"DeepSeek 工具参数不是合法 JSON: {raw_arguments}"
                    ) from exc

                if name == "finalize_documents" and forced_chunk_mode == "metadata_only":
                    arguments["chunk_mode"] = "metadata_only"
                    arguments["detail_queries"] = []

                if progress_callback:
                    if name == "soft_search_documents":
                        progress_callback("正在查找相关文件…")
                    elif name == "hard_filter_documents":
                        progress_callback("正在筛选相关文件…")
                    elif name == "sort_documents":
                        progress_callback("正在整理文件顺序…")
                    elif name == "finalize_documents":
                        if arguments.get("chunk_mode") == "relevant":
                            progress_callback("正在规划正文检索…")
                        elif arguments.get("chunk_mode") == "all":
                            progress_callback("正在准备读取完整文档…")
                        else:
                            progress_callback("正在整理文件信息…")

                result = execute_tool(name, arguments)
                tool_trace.append(
                    {
                        "tool": name,
                        "arguments": arguments,
                        "result": result,
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

                if name == "finalize_documents":
                    return {
                        "tool_trace": tool_trace,
                        "final": result,
                    }

        raise RuntimeError("DeepSeek 文档调度器超过最大工具调用轮数，仍未确定文件。")

    @staticmethod
    def _build_evidence(sources: list[dict[str, Any]]) -> str:
        blocks = []
        for source in sources:
            header = {
                "citation_id": source["citation_id"],
                "document_id": source.get("document_id"),
                "file_hash": source.get("file_hash"),
                "file_name": source.get("file_name"),
                "source_path": source.get("source_path"),
                "project_name": source.get("project_name"),
                "document_type": source.get("document_type"),
                "version": source.get("version"),
                "document_date": source.get("document_date"),
                "source_created_at": source.get("source_created_at"),
                "source_modified_at": source.get("source_modified_at"),
                "location": source.get("location"),
            }
            blocks.append(
                f"[{source['citation_id']}]\n"
                f"metadata: {json.dumps(header, ensure_ascii=False)}\n"
                f"content:\n{source['content']}"
            )
        return "\n\n--------------------\n\n".join(blocks)

    @staticmethod
    def _document_manifest(documents: list[dict[str, Any]]) -> str:
        manifest = []
        for document in documents:
            metadata = document.get("metadata") or {}
            manifest.append(
                {
                    "document_id": document.get("document_id"),
                    "file_hash": metadata.get("file_hash"),
                    "file_name": metadata.get("file_name"),
                    "source_path": metadata.get("source_path"),
                    "project_name": metadata.get("project_name"),
                    "document_type": metadata.get("document_type"),
                    "version": metadata.get("version"),
                    "document_date": metadata.get("document_date"),
                    "source_created_at": metadata.get("source_created_at"),
                    "source_modified_at": metadata.get("source_modified_at"),
                }
            )
        return json.dumps(manifest, ensure_ascii=False, indent=2)

    def answer_general(self, question: str) -> str:
        system_prompt = """你是通用问答助手。
当前问题已经被前置路由判定为不需要访问公司内部知识库。
请直接依据通用知识回答用户问题。

要求：
1. 不要声称你查询了公司资料、数据库、内部文件或知识库。
2. 普通聊天就自然聊天；技术问题就直接解释技术知识。
3. 如果用户的问题本身要求公司内部事实，而现有对话没有提供这些事实，不要编造公司内部信息。
4. 回答清楚、自然、不过度展开。
"""
        data = self._post(
            {
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
            }
        )
        return data["choices"][0]["message"]["content"].strip()

    def answer_metadata(
        self,
        question: str,
        documents: list[dict[str, Any]],
        duplicate_groups: list[dict[str, Any]] | None = None,
    ) -> str:
        system_prompt = """你是企业知识库的文档 metadata 回答助手。
只能依据系统提供的最终文件清单和重复文件信息回答，不读取也不假装读取正文。

必须遵守：
1. 只回答文件级属性，例如文件名、版本、日期、路径、文件类型、数量等。
2. file_name 必须逐字使用清单中的真实字符串，不得把相似文件名说成相同文件名。
3. document_id 是文件记录的唯一身份；document_id 不同就保留为不同文件记录。
4. file_hash 相同表示文件内容字节完全相同；file_hash 不同表示内容不同。只有系统提供的 hash 能支持这个判断。
5. 同名同版本文件若 file_hash 不同，要明确指出存在多份内容不同的文件；若 hash 相同，则说明是内容相同的重复文件记录/副本。
6. 用户要求“只罗列文件名”时，就只列文件名，不扩写正文内容。
7. metadata 缺失就说缺失，不从文件名或常识补全。
"""
        user_prompt = f"""用户问题：
{question}

最终文件清单：
{self._document_manifest(documents)}

同名同版本重复文件信息：
{json.dumps(duplicate_groups or [], ensure_ascii=False, indent=2)}

请只根据上述 metadata 回答。"""

        data = self._post(
            {
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
        )
        return data["choices"][0]["message"]["content"].strip()

    def answer(
        self,
        question: str,
        sources: list[dict[str, Any]],
        documents: list[dict[str, Any]],
        duplicate_groups: list[dict[str, Any]] | None = None,
    ) -> str:
        if not sources:
            return "已找到相关文件，但这些文件中没有可用于回答的正文内容。"

        system_prompt = """你是企业工业项目知识库问答助手。
你只能依据系统提供的已选中文件和证据正文回答，不能把模型自身知识当作公司内部事实。

必须遵守：
1. 每一个基于公司资料的关键事实，都在对应句子后标注引用，例如 [S1]。
2. 只能使用证据中实际存在的引用编号，禁止编造引用。
3. 不要编造文件名、版本、日期、客户、技术参数或报价。
4. 证据不足时明确说“现有资料无法确认”。
5. document_id 才是文件唯一身份。即使 file_name 和 version 完全相同，只要 document_id 不同，就必须视为不同文件，不能擅自合并。
6. 只有在系统重复文件信息明确显示同名同版本时，才可以称为“同名同版本”。其中 file_hash 相同表示内容相同，file_hash 不同表示内容不同；不得仅凭文件名看起来相似就自行判断。若同名同版本且内容不同，必须分别说明差异，不要替用户决定哪份才是“正确版本”。
7. 比较多个文件时，应逐文件、逐章节/同标题内容进行对照。若系统提供的是完整文档 chunk，不得只比较最显眼的几个片段而忽略其余内容。
8. 最终事实只能由对应证据块的 content 正文直接支持。metadata 只用于识别文件和定位来源，不能代替正文成为技术事实依据。
9. summary、keywords 等检索辅助 metadata 不得作为回答事实。
10. 文件之间存在冲突时分别列出，不擅自裁决。
11. 回答应直接、清楚，不需要自己生成“引用来源”总表，系统会自动追加。
"""

        user_prompt = f"""用户问题：
{question}

本次已经确定使用的文件清单：
{self._document_manifest(documents)}

检测到的同名同版本多文件组：
{json.dumps(duplicate_groups or [], ensure_ascii=False, indent=2)}

以下是这些文件提供的证据正文：

{self._build_evidence(sources)}

请严格依据上述文件与正文回答。"""

        data = self._post(
            {
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
        )
        return data["choices"][0]["message"]["content"].strip()
