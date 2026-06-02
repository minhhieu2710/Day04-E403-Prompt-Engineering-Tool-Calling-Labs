import json
import logging
from typing import List

from langchain_core.tools import tool

from utils.data_store import OrderDataStore
from core.schemas import (
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    OrderLineInput,
    ProductDetailInput,
    SaveOrderInput,
)
from .service import OrderService

logger = logging.getLogger(__name__)


def build_tools(store: OrderDataStore):
    """Create tool functions that delegate to OrderService.

    Returns a list of tool callables ready for ``create_agent``.
    """
    service = OrderService(store)

    @tool(description="Use when you need to search the product catalog. Must be called first after all customer info is collected.")
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: List[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        logger.info("Tool list_products invoked")
        payload = service.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags,
            in_stock_only=in_stock_only,
            limit=limit,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(description="Retrieve detailed product information after list_products. Returns a detail_token needed for later calculations.")
    def get_product_details(product_ids: List[str]) -> str:
        logger.info("Tool get_product_details invoked for %s", product_ids)
        payload = service.get_product_details(product_ids)
        return json.dumps(payload, ensure_ascii=False)

    @tool(description="Get the discount campaign for the order. Must be called after product details.")
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        logger.info("Tool get_discount invoked with seed_hint=%s", seed_hint)
        payload = service.get_discount(seed_hint=seed_hint, customer_tier=customer_tier)
        return json.dumps(payload, ensure_ascii=False)

    @tool(description="Calculate final totals using detail_token and discount. Must follow get_discount.")
    def calculate_order_totals(
        items: List[OrderLineInput],
        detail_token: str,
        discount_rate: float,
    ) -> str:
        logger.info("Tool calculate_order_totals invoked")
        payload = service.calculate_order_totals(items=items, detail_token=detail_token, discount_rate=discount_rate)
        return json.dumps(payload, ensure_ascii=False)

    @tool(description="Persist the completed order. Must be the final step.")
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items: List[OrderLineInput],
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        logger.info("Tool save_order invoked for %s", customer_name)
        result = service.save_order(
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
        return json.dumps(result, ensure_ascii=False)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]
