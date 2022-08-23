from ffcv.loader import ORDER_TYPE, OrderOption


def get_order_option(order: str) -> ORDER_TYPE:
    if order == "random":
        order_option = OrderOption.RANDOM
    elif order == "quasi_random":
        order_option = OrderOption.QUASI_RANDOM
    elif order == "sequential":
        order_option = OrderOption.SEQUENTIAL
    else:
        raise ValueError(f"Unknown order option: {order}")
    return order_option
