from statplotannot.plots import adjust_lightness


def colors_2arm(amount=1):
    # return [
    #     adjust_lightness("#E89317", amount=amount),  # Unstruc
    #     adjust_lightness("#3980ea", amount=amount),  # Struc
    # ]
    return [
        # adjust_lightness("#f55673", amount=amount),  # Unstruc
        # adjust_lightness("#f72585", amount=amount),  # Unstruc
        # adjust_lightness("#3baaa1", amount=amount),  # Struc
        adjust_lightness("#Ff7518", amount=amount),  # Unstruc
        adjust_lightness("#008000", amount=amount),  # Struc
    ]


def colors_2arm_swap(amount=1):
    return [
        adjust_lightness("#f55673", amount=amount),  # Unstruc
        adjust_lightness("#ea8c67", amount=amount),  # Unstruc on struc
        adjust_lightness("#3baaa1", amount=amount),  # Struc
        adjust_lightness("#4d89cd", amount=amount),  # Struc on unstruc
    ]
