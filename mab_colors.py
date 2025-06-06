from statplotannot.plots import adjust_lightness


def colors_2arm(amount=1):
    # return [
    #     adjust_lightness("#E89317", amount=amount),  # Unstruc
    #     adjust_lightness("#3980ea", amount=amount),  # Struc
    # ]
    return [
        adjust_lightness("#f55673", amount=amount),  # Unstruc
        # adjust_lightness("#f72585", amount=amount),  # Unstruc
        adjust_lightness("#3baaa1", amount=amount),  # Struc
    ]
