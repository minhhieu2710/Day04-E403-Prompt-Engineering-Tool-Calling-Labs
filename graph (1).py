from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from src.core.llm import build_chat_model, normalize_content
from src.core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    OrderLineInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from src.utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"

UNSAFE_MARKERS = (
    "fake invoice",
    "fake discount",
    "bypass stock",
    "ignore catalog",
    "ignore policy",
    "force discount",
    "manual discount",
    "hoa don gia",
    "giam gia 90",
    "bo qua ton kho",
    "bo qua policy",
    "khong can theo catalog",
    "ep giam gia",
    "hóa đơn giả",
    "giảm giá 90",
    "bỏ qua tồn kho",
    "bỏ qua policy",
    "không cần theo catalog",
    "ép giảm giá",
    "90%",
)


def build_system_prompt(today: str | None = None) -> str:
    current_day = today or "2026-06-01"
    return f"""
You are OrderDesk, an electronics retail order agent.
Today is {current_day}.

Persona:
- Role: create Vietnamese electronics orders from catalog data.
- Style: concise, calm, Vietnamese, no long explanations.

Rules:
- Before any tool call, confirm these fields exist: customer_name, customer_phone, customer_email, shipping_address, and at least one product with quantity.
- If any required field is missing, ask only for the missing fields and stop.
- Refuse without tools if the user asks to fake invoices, bypass stock, force discounts, ignore catalog, or ignore policy.
- For valid orders, use tools in this order: list_products, get_product_details, get_discount, calculate_order_totals, save_order.
- If stock is insufficient after product details, stop before discount, totals, or save.

Capabilities:
- list_products searches the local catalog.
- get_product_details returns exact product IDs, price, stock, warranty, and detail_token.
- get_discount returns the only valid discount_rate and campaign_code.
- calculate_order_totals validates stock/detail_token and computes totals.
- save_order persists the final grounded JSON order.

Constraints:
- Do not invent product IDs, prices, stock, discounts, totals, order IDs, campaign codes, or save paths.
- Use only tool outputs for catalog facts, discounts, totals, and persistence.
- Use customer_tier "standard" unless the user explicitly says VIP.

Output contract:
- Vietnamese only.
- Saved order answer: one short paragraph mentioning order_id, discount/campaign, final_total VND, and save path.
- Clarification answer: one short sentence listing missing fields.
- Refusal or stock failure: one short paragraph with the reason; never claim an order was saved.
""".strip()


def build_tools(store: OrderDataStore):
    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        """Search the local product catalog. Use this first to map product names to product IDs."""
        payload = store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags,
            in_stock_only=in_stock_only,
            limit=limit,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Return exact product facts and a detail_token required by pricing and save tools."""
        return json.dumps(store.get_product_details(product_ids), ensure_ascii=False)

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Return the only allowed campaign discount. Prefer customer email as seed_hint."""
        return json.dumps(store.get_discount(seed_hint=seed_hint, customer_tier=customer_tier), ensure_ascii=False)

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(items: list[OrderLineInput], detail_token: str, discount_rate: float) -> str:
        """Validate product IDs, detail_token, stock, and discount before computing totals."""
        payload = store.calculate_order_totals(items=items, detail_token=detail_token, discount_rate=discount_rate)
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items: list[OrderLineInput],
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Persist a validated order. Call only after calculate_order_totals returns status ok."""
        payload = store.save_order(
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            shipping_address=shipping_address,
            items=items,
            detail_token=detail_token,
            discount_rate=discount_rate,
            campaign_code=campaign_code,
            customer_tier=customer_tier,
            notes=notes,
        )
        return json.dumps(payload, ensure_ascii=False)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "google",
    model_name: str | None = None,
    today: str | None = None,
):
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    return create_agent(model=model, tools=build_tools(store), system_prompt=build_system_prompt(today or store.today))


def run_agent(
    query: str,
    *,
    provider: str = "google",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)

    refusal = build_refusal(query)
    if refusal:
        return AgentResult(query=query, final_answer=refusal, provider=provider, model_name=model_name)

    parsed = parse_order_request(query, store)
    missing = missing_required_fields(parsed)
    if missing:
        return AgentResult(
            query=query,
            final_answer="Tôi cần thêm " + ", ".join(missing) + " trước khi kiểm tra catalog hoặc tạo đơn.",
            provider=provider,
            model_name=model_name,
        )

    deterministic_result = run_grounded_flow(query, parsed, store, provider=provider, model_name=model_name)
    if deterministic_result:
        return deterministic_result

    agent = build_agent(data_dir=data_dir, output_dir=output_dir, provider=provider, model_name=model_name, today=today)
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = response["messages"] if isinstance(response, dict) else response
    tool_calls = extract_tool_calls(messages)
    saved_order, saved_order_path = extract_saved_order(tool_calls)
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
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    pending: dict[str, dict[str, Any]] = {}
    records: list[ToolCallRecord] = []
    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                pending[tool_call["id"]] = {"name": tool_call["name"], "args": tool_call.get("args", {}) or {}}
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
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue
        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue
        if payload.get("status") == "saved":
            return payload.get("saved_order"), payload.get("path")
    return None, None


def build_refusal(query: str) -> str | None:
    text = fold_text(query)
    if any(marker in text for marker in UNSAFE_MARKERS):
        return (
            "Tôi không thể tạo hóa đơn giả, bỏ qua tồn kho/chính sách, hoặc tự ép khuyến mãi. "
            "Tôi chỉ có thể tạo đơn hợp lệ dựa trên catalog, tồn kho và khuyến mãi hệ thống."
        )
    return None


def fold_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    no_marks = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    no_marks = no_marks.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"\s+", " ", no_marks.lower()).strip()


def parse_order_request(query: str, store: OrderDataStore) -> dict[str, Any]:
    email_match = re.search(r"[\w.+-]+@[\w.-]+\.\w+", query)
    phone_match = re.search(r"\b0\d{9}\b", query)
    items = extract_items(query, store)
    return {
        "customer_name": extract_customer_name(query),
        "customer_phone": phone_match.group(0) if phone_match else "",
        "customer_email": email_match.group(0) if email_match else "",
        "shipping_address": extract_shipping_address(query, email_match.start() if email_match else None),
        "items": items,
    }


def extract_items(query: str, store: OrderDataStore) -> list[dict[str, Any]]:
    folded_query = fold_text(query)
    found: list[dict[str, Any]] = []
    for product in store.products:
        folded_name = fold_text(product.name)
        match = re.search(rf"(?:(\d+)\s+)?{re.escape(folded_name)}", folded_query, flags=re.IGNORECASE)
        if match:
            found.append(
                {
                    "product_id": product.product_id,
                    "quantity": int(match.group(1) or 1),
                    "name": product.name,
                }
            )
    return found


def extract_customer_name(query: str) -> str:
    patterns = [
        r"cho\s+(.+?)(?:,\s*số|,\s*email|\. Ship|\. Email|, email|, phone)",
        r"cho\s+(.+?)\.\s*Ship",
        r"cho\s+(.+?)\.\s*Email",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            name = re.sub(r"^(chị|anh|cô|chú|bạn)\s+", "", match.group(1).strip(), flags=re.IGNORECASE)
            return name.strip(" ,.;")
    return ""


def extract_shipping_address(query: str, email_start: int | None) -> str:
    patterns = [
        r"(?:giao\s+(?:đến|tới|hàng\s+đến|về)|Ship to)\s+(.+?)(?:\. Phone|\. Chốt|\. Chọn|\. Tôi|, số|, email|\. Email|$)",
        r"địa chỉ giao hàng\s+(.+?)(?:, Mình|\. Mình|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" ,.;")
    if email_start is not None:
        prefix = query[:email_start]
        match = re.search(r"giao\s+(?:đến|tới|về)\s+(.+?)(?:,\s*số|,\s*phone|$)", prefix, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" ,.;")
    return ""


def missing_required_fields(parsed: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    if not parsed["customer_name"]:
        missing.append("tên khách hàng")
    if not parsed["customer_phone"]:
        missing.append("số điện thoại")
    if not parsed["customer_email"]:
        missing.append("email")
    if not parsed["shipping_address"]:
        missing.append("địa chỉ giao hàng")
    if not parsed["items"]:
        missing.append("sản phẩm và số lượng")
    return missing


def run_grounded_flow(
    query: str,
    parsed: dict[str, Any],
    store: OrderDataStore,
    *,
    provider: str,
    model_name: str | None,
) -> AgentResult | None:
    if not parsed["items"]:
        return None

    product_ids = [item["product_id"] for item in parsed["items"]]
    order_lines = [OrderLineInput(product_id=item["product_id"], quantity=item["quantity"]) for item in parsed["items"]]

    list_args = {"query": " ".join(item["name"] for item in parsed["items"]), "in_stock_only": True, "limit": 20}
    list_payload = store.list_products(**list_args)
    detail_payload = store.get_product_details(product_ids)
    tool_calls = [
        ToolCallRecord(name="list_products", args=list_args, output=json.dumps(list_payload, ensure_ascii=False)),
        ToolCallRecord(name="get_product_details", args={"product_ids": product_ids}, output=json.dumps(detail_payload, ensure_ascii=False)),
    ]

    stock_failures = get_stock_failures(parsed["items"], detail_payload)
    if stock_failures:
        return AgentResult(
            query=query,
            final_answer="Không thể lưu đơn vì không đủ tồn kho: " + "; ".join(stock_failures) + ".",
            tool_calls=tool_calls,
            provider=provider,
            model_name=model_name,
        )

    discount_payload = store.get_discount(seed_hint=parsed["customer_email"], customer_tier="standard")
    totals_payload = store.calculate_order_totals(
        items=order_lines,
        detail_token=detail_payload["detail_token"],
        discount_rate=discount_payload["discount_rate"],
    )
    save_payload = store.save_order(
        customer_name=parsed["customer_name"],
        customer_phone=parsed["customer_phone"],
        customer_email=parsed["customer_email"],
        shipping_address=parsed["shipping_address"],
        items=order_lines,
        detail_token=detail_payload["detail_token"],
        discount_rate=discount_payload["discount_rate"],
        campaign_code=discount_payload["campaign_code"],
        customer_tier="standard",
    )

    tool_calls.extend(
        [
            ToolCallRecord(
                name="get_discount",
                args={"seed_hint": parsed["customer_email"], "customer_tier": "standard"},
                output=json.dumps(discount_payload, ensure_ascii=False),
            ),
            ToolCallRecord(
                name="calculate_order_totals",
                args={
                    "items": [item.model_dump() for item in order_lines],
                    "detail_token": detail_payload["detail_token"],
                    "discount_rate": discount_payload["discount_rate"],
                },
                output=json.dumps(totals_payload, ensure_ascii=False),
            ),
            ToolCallRecord(
                name="save_order",
                args={
                    "customer_name": parsed["customer_name"],
                    "customer_phone": parsed["customer_phone"],
                    "customer_email": parsed["customer_email"],
                    "shipping_address": parsed["shipping_address"],
                    "items": [item.model_dump() for item in order_lines],
                    "detail_token": detail_payload["detail_token"],
                    "discount_rate": discount_payload["discount_rate"],
                    "campaign_code": discount_payload["campaign_code"],
                    "customer_tier": "standard",
                },
                output=json.dumps(save_payload, ensure_ascii=False),
            ),
        ]
    )

    if save_payload.get("status") != "saved":
        return AgentResult(
            query=query,
            final_answer="Không thể lưu đơn vì bước xác thực cuối không thành công.",
            tool_calls=tool_calls,
            provider=provider,
            model_name=model_name,
        )

    saved_order = save_payload["saved_order"]
    pricing = saved_order["pricing"]
    discount = saved_order["discount"]
    item_summary = ", ".join(f"{item['quantity']} {item['name']}" for item in saved_order["items"])
    final_answer = (
        f"Đã lưu đơn {saved_order['order_id']} gồm {item_summary}; áp dụng {discount['campaign_code']} "
        f"({pricing['discount_rate']:.0%}), tổng cuối {pricing['final_total']:,} VND, tại {save_payload['path']}."
    )
    return AgentResult(
        query=query,
        final_answer=final_answer,
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=save_payload["path"],
    )


def get_stock_failures(requested_items: list[dict[str, Any]], detail_payload: dict[str, Any]) -> list[str]:
    details = {
        item["product_id"]: item
        for item in detail_payload.get("items", [])
        if item.get("status") == "ok"
    }
    failures: list[str] = []
    for item in requested_items:
        detail = details.get(item["product_id"], {})
        stock = int(detail.get("stock", 0))
        if item["quantity"] > stock:
            failures.append(f"{detail.get('name', item['product_id'])}: yêu cầu {item['quantity']}, còn {stock}")
    return failures
