import logging
from typing import Any, List

from utils.data_store import OrderDataStore
from core.schemas import (
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    OrderLineInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)

logger = logging.getLogger(__name__)


class OrderService:
    """Service layer encapsulating business logic and data store access.

    Provides thin wrappers around ``OrderDataStore`` methods and centralises
    validation, logging and error handling. This makes the tool functions
    straightforward and improves testability.
    """

    def __init__(self, store: OrderDataStore) -> None:
        self.store = store
        logger.debug("OrderService initialised with data_dir=%s", store.data_dir)

    # ----- Tool helper methods -----
    def list_products(self, query: str | None = None, category: str | None = None,
                     max_unit_price: int | None = None, required_tags: List[str] | None = None,
                     in_stock_only: bool = True, limit: int = 8) -> Any:
        logger.info("list_products called with query=%s", query)
        return self.store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags or [],
            in_stock_only=in_stock_only,
            limit=limit,
        )

    def get_product_details(self, product_ids: List[str]) -> Any:
        logger.info("get_product_details for %s", product_ids)
        return self.store.get_product_details(product_ids)

    def get_discount(self, seed_hint: str, customer_tier: str = "standard") -> Any:
        logger.info("get_discount with seed_hint=%s, tier=%s", seed_hint, customer_tier)
        return self.store.get_discount(seed_hint=seed_hint, customer_tier=customer_tier)

    def calculate_order_totals(self, items: List[OrderLineInput], detail_token: str,
                             discount_rate: float) -> Any:
        logger.info("calculate_order_totals called")
        return self.store.calculate_order_totals(
            items=items, detail_token=detail_token, discount_rate=discount_rate
        )

    def save_order(
        self,
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
    ) -> Any:
        logger.info("save_order called for customer %s", customer_name)
        return self.store.save_order(
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
