from ffcv.loader import OrderOption


def get_order_option_ffcv(order):
    if order == "random":
        order_option = OrderOption.RANDOM
    elif order == "quasi_random":
        order_option = OrderOption.QUASI_RANDOM
    elif order == "sequential":
        order_option = OrderOption.SEQUENTIAL
    else:
        raise ValueError(f"Unknown order option: {order}")
    return order_option
