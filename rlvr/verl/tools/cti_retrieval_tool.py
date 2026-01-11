"""Custom CTI retrieval tool for TARBA."""

from __future__ import annotations

import json
from typing import Any, Optional

import requests

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionParametersSchema, OpenAIFunctionPropertySchema, OpenAIFunctionSchema, OpenAIFunctionToolSchema, ToolResponse


class CTIRetrievalTool(BaseTool):
    """Tool wrapper that queries the TARBA retrieval server."""

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        if tool_schema is None:
            tool_schema = self._default_schema()
        super().__init__(config=config, tool_schema=tool_schema)
        self.retrieval_url = config.get("retrieval_url")
        if not self.retrieval_url:
            raise ValueError("retrieval_url is required for CTIRetrievalTool")
        self.timeout = float(config.get("timeout", 30))
        self.topk_cap = int(config.get("topk_cap", 8))
        self.max_query_chars = int(config.get("max_query_chars", 256))
        self.max_snippet_chars = int(config.get("max_snippet_chars", 800))

    def _default_schema(self) -> OpenAIFunctionToolSchema:
        params = OpenAIFunctionParametersSchema(
            type="object",
            properties={
                "query": OpenAIFunctionPropertySchema(type="string", description="Keyword query"),
                "topk": OpenAIFunctionPropertySchema(type="integer", description="Number of docs to return"),
            },
            required=["query", "topk"],
        )
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="cti_retrieve",
                description="Retrieve CTI label docs by keyword query.",
                parameters=params,
            ),
        )

    def _truncate(self, text: str) -> str:
        if self.max_snippet_chars <= 0:
            return text
        if len(text) <= self.max_snippet_chars:
            return text
        return text[: self.max_snippet_chars].rstrip() + "..."

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        query = str(parameters.get("query") or "")[: self.max_query_chars]
        label_type = str(kwargs.get("label_type") or "").strip()
        if not label_type:
            label_type = str(parameters.get("label_type") or "").strip()
        topk_raw = parameters.get("topk", self.topk_cap)
        try:
            topk = int(topk_raw)
        except (TypeError, ValueError):
            topk = self.topk_cap
        topk = min(max(topk, 1), self.topk_cap)
        budget_B = kwargs.get("budget_B")
        if budget_B is not None:
            try:
                budget_B = int(budget_B)
            except (TypeError, ValueError):
                budget_B = None
        if budget_B is not None:
            if budget_B <= 0:
                return (
                    ToolResponse(text="DOC_IDS: []\nRetrieval disabled for this sample."),
                    0.0,
                    {"doc_ids": [], "topk": 0},
                )
            topk = min(topk, budget_B)

        if not query or not label_type:
            return (
                ToolResponse(text="DOC_IDS: []\nNo query or label_type provided."),
                0.0,
                {"doc_ids": [], "topk": 0},
            )

        payload = {
            "label_type": label_type,
            "query": query,
            "topk": topk,
            "return_scores": True,
        }

        try:
            resp = requests.post(self.retrieval_url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            msg = f"DOC_IDS: []\nRetrieval error: {exc}"
            return ToolResponse(text=msg), 0.0, {"doc_ids": [], "topk": 0}

        results = data.get("results", []) if isinstance(data, dict) else []
        doc_ids = []
        lines = []
        for idx, item in enumerate(results, start=1):
            if not isinstance(item, dict):
                continue
            doc_id = str(item.get("doc_id") or "").strip()
            title = str(item.get("title") or "").strip()
            snippet = str(item.get("snippet") or "").strip()
            if doc_id:
                doc_ids.append(doc_id)
            snippet = self._truncate(snippet)
            lines.append(f"{idx}) {doc_id} - {title}")
            if snippet:
                lines.append(f"   {snippet}")

        header = f"DOC_IDS: {json.dumps(doc_ids)}"
        text = "\n".join([header] + lines) if lines else header
        return ToolResponse(text=text), 0.0, {"doc_ids": doc_ids, "topk": int(topk)}


__all__ = ["CTIRetrievalTool"]
