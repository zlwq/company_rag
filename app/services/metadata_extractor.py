import json
import os
from datetime import datetime, timezone
from typing import Any

import requests


METADATA_FIELDS = {
    "project_name": None,
    "document_type": None,
    "version": None,
    "project_stage": None,
    "customer_name": None,
    "document_date": None,
    "keywords": [],
    "summary": None,
}


class MetadataExtractor:
    def __init__(self) -> None:
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.timeout = int(os.getenv("DEEPSEEK_TIMEOUT", "60"))
        self.max_chars = int(os.getenv("METADATA_MAX_CHARS", "16000"))

    def _document_excerpt(self, parsed_document: dict[str, Any]) -> str:
        text = "\n\n".join(
            unit["content"] for unit in parsed_document["units"] if unit.get("content")
        )

        if len(text) <= self.max_chars:
            return text

        head_size = int(self.max_chars * 0.75)
        tail_size = self.max_chars - head_size
        return text[:head_size] + "\n\n...[中间内容已省略]...\n\n" + text[-tail_size:]

    def empty_metadata(self) -> dict[str, Any]:
        result = dict(METADATA_FIELDS)
        result["keywords"] = []
        result["metadata_source"] = "none"
        result["metadata_extracted_at"] = datetime.now(timezone.utc).isoformat()
        return result

    def extract(self, parsed_document: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            return self.empty_metadata()

        system_prompt = """你负责从企业工业项目文档中提取文档级 metadata。
只能依据文件名、文件属性和提供的正文内容填写字段；无法确认的字段返回 null，不得凭空补全。

字段规则：
1. project_name：提取实际项目、设备、试验台或系统名称。若文件名明确包含项目名称，可直接从文件名提取。例如“蓄压器综合试验台设计方案-某公司-20261008-V3.5.docx”的 project_name 应为“蓄压器综合试验台”，而不是年份、公司名或版本号。
2. document_type：若文件名或正文明确出现“设计方案、技术方案、报价、报价单、技术协议、合同、验收报告、说明书、会议纪要”等类型，应填写对应类型。
3. version：只有文件名或正文明确出现 V1、V3.5、Rev.A 等版本信息时填写。
4. project_stage：只有明确出现“草案、评审版、送审版、最终版、正式版、验收版”等阶段信息时填写。
5. customer_name：只有明确能确认客户/甲方名称时填写，不要把编制单位自动当成客户。
6. document_date：文件名或正文中明确出现文档日期时填写，优先输出 YYYY-MM-DD。像 20261008 可解析为 2026-10-08。文件系统修改时间不能当作文档日期。
7. keywords：返回 3 到 10 个简短关键词；信息不足可返回空数组。
8. summary：用一句到三句话概括文档内容；正文为空则 null。

只输出 JSON 对象，不要输出解释文字。"""

        payload_for_llm = {
            "file_name": parsed_document["file_name"],
            "file_type": parsed_document["file_type"],
            "file_size": parsed_document.get("file_size"),
            "source_modified_at": parsed_document.get("source_modified_at"),
            "source_created_at": parsed_document.get("source_created_at"),
            "content": self._document_excerpt(parsed_document),
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(payload_for_llm, ensure_ascii=False),
                    },
                ],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:].lstrip()

        raw = json.loads(content)
        result = dict(METADATA_FIELDS)
        result["keywords"] = []

        for key in METADATA_FIELDS:
            value = raw.get(key)
            if key == "keywords":
                if isinstance(value, list):
                    result[key] = [
                        str(item).strip()
                        for item in value
                        if str(item).strip()
                    ][:10]
            elif value is None or isinstance(value, str):
                result[key] = (
                    value.strip()
                    if isinstance(value, str) and value.strip()
                    else None
                )

        result["metadata_source"] = "deepseek"
        result["metadata_extracted_at"] = datetime.now(timezone.utc).isoformat()
        return result
