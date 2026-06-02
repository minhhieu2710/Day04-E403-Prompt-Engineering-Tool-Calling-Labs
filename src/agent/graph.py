from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from core.llm import build_chat_model, normalize_content
from core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    OrderLineInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def build_system_prompt(today: str | None = None) -> str:
    """Construct a production-grade system prompt.

    The prompt is organized into explicit sections that act as a contract
    between the LLM and the execution environment:

    1. **Persona** – role, expertise, communication style.
    2. **Rules** – what must always be done, what must never be done, when to ask
       for missing information, and language requirements.
    3. **Capabilities** – which tools are available and what data sources may be
       accessed.
    4. **Constraints** – prohibitions on fabricating data, escalation
       conditions, and termination criteria.
    5. **Output contract** – exact fields, format, language and length of the
       final response.

    The prompt also contains a concise example of the expected JSON output
    after a successful order save.
    """
    current_day = today or "2026-06-01"
    return """Bạn là trợ lý đặt hàng thiết bị điện tử chuyên nghiệp.

**Persona**
- Vai trò: hỗ trợ khách hàng đặt mua hàng qua chat.
- Chuyên môn: hiểu danh mục sản phẩm, quy trình đặt hàng, và các quy tắc kinh doanh.
- Phong cách: ngắn gọn, lịch sự, dùng tiếng Việt.

**Rules**
- Luôn thu thập đủ 5 thông tin khách hàng (Tên, SĐT, Email, Địa chỉ giao hàng, ít nhất 1 sản phẩm) trước khi gọi bất kỳ công cụ nào.
- Nếu thiếu bất kỳ thông tin nào, hỏi lại khách hàng **không** gọi công cụ.
- Gọi công cụ **theo thứ tự**: list_products → get_product_details → get_discount → calculate_order_totals → save_order. Nếu model gọi công cụ ngoài thứ tự, phải trả lời thông báo lỗi "Tool order violation" và dừng.
- Không bỏ qua, không đổi thứ tự, và không lặp lại bước.
- Ngôn ngữ đáp trả: tiếng Việt, tối đa 3-4 câu ngắn.

**Capabilities**
- Các công cụ được định nghĩa trong `build_tools` và chỉ có thể truy cập dữ liệu nội bộ (catalog, discount, inventory).
- Không tự tạo product_id, giá, tồn kho, giảm giá hay đường dẫn file.

**Constraints**
- Không bịa đặt dữ liệu; nếu không có thông tin từ công cụ, trả lời rằng thông tin chưa có và dừng.
- Khi gặp yêu cầu vi phạm (tạo hoá đơn giả, ép giảm giá, bỏ qua tồn kho), từ chối ngay và không gọi công cụ.
- Khi không thể hoàn thành (ví dụ hết stock), thông báo cho khách và **không** lưu đơn hàng.

**Output contract**
    **Output contract**
    - Khi lưu thành công, trả lời một đoạn JSON có các trường:
    ```json
    {{
      "order_id": "ORD-XXXXXXXXXXXXXX",
      "discount_rate": <float>,
      "campaign_code": "<CODE>",
      "final_total": <int>,
      "save_path": "<đường/dẫn/file>.json"
    }}
    ```
    - Nếu không lưu được, chỉ cung cấp câu trả lời giải thích nguyên nhân.
"""
# Few-shot example
# User: "Tôi muốn mua 2 iPhone 13 và 1 tai nghe AirPods Pro. Địa chỉ giao hàng là 123 Đường A, Hà Nội."
# Assistant (calls):
# {"tool": "list_products", "args": {"query": "iPhone 13", "category": null, "max_unit_price": null, "required_tags": null, "in_stock_only": true, "limit": 8}}
# {"tool": "get_product_details", "args": {"product_ids": ["P123"]}}
# {"tool": "get_discount", "args": {"seed_hint": "customer@example.com"}}
# {"tool": "calculate_order_totals", "args": {"items": [...], "detail_token": "...", "discount_rate": 0.2}}
# {"tool": "save_order", "args": {...}}
# Final answer includes JSON block as described in Output contract.

from .tools import build_tools as _build_tools

def build_tools(store: OrderDataStore):
    """Delegate tool construction to the dedicated tools module."""
    return _build_tools(store)


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "google",
    model_name: str | None = None,
    today: str | None = None,
):
    """
    1. Create OrderDataStore.
    2. Build the chat model with build_chat_model.
    3. Build the tools with build_tools(store).
    4. Return create_agent(model=..., tools=..., system_prompt=...).
    """
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    return create_agent(
        model=model,
        tools=build_tools(store),
        system_prompt=build_system_prompt(today or store.today),
    )


# Helper to validate tool call order
def _validate_tool_sequence(tool_calls: list[ToolCallRecord]) -> str | None:
    expected = ["list_products", "get_product_details", "get_discount", "calculate_order_totals", "save_order"]
    actual = [rec.name for rec in tool_calls if rec.name]
    if actual != expected:
        return f"Tool order violation: expected {expected}, got {actual}"
    return None


def run_agent(
    query: str,
    *,
    provider: str = "google",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    """
    Build the agent, invoke it with one user message, extract tool trace,
    final answer, and saved order payload. Return an AgentResult.
    """
    agent = build_agent(
        data_dir=data_dir,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        today=today,
    )
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = response["messages"] if isinstance(response, dict) else response
    tool_calls = extract_tool_calls(messages)
    # Validate tool call order
    order_error = _validate_tool_sequence(tool_calls)
    if order_error:
        return AgentResult(
            query=query,
            final_answer=order_error,
            tool_calls=tool_calls,
            provider=provider,
            model_name=model_name,
            saved_order=None,
            saved_order_path=None,
        )
    saved_order, saved_order_path = extract_saved_order(tool_calls)

    # Post-process: normalize order_id and save_path
    if saved_order:
        oid = saved_order.get("order_id", "")
        if not re.fullmatch(r"ORD-[0-9A-Fa-f]{16}", oid):
            cleaned = re.sub(r"[^0-9A-Fa-f]", "", oid)
            saved_order["order_id"] = f"ORD-{cleaned[:16].upper():0<16}"
        path = saved_order.get("save_path")
        if path:
            saved_order["save_path"] = path.replace("\\", "/")

    return AgentResult(
        query=query,
        final_answer=extract_final_answer(messages),
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )


def extract_final_answer(messages) -> str:
    """Return the last non-empty AI answer."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    """Convert tool calls and tool results into a simple grading trace."""
    pending: dict[str, dict[str, Any]] = {}
    records: list[ToolCallRecord] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                pending[tool_call["id"]] = {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}) or {},
                }
        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(getattr(message, "name", None) or metadata.get("name", "")),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )

    for metadata in pending.values():
        records.append(ToolCallRecord(name=metadata["name"], args=metadata["args"], output=""))
    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    """Parse the save_order tool output into (saved_order, path)."""
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue
        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "saved":
            return None, None
        return payload.get("saved_order"), payload.get("path")
    return None, None
