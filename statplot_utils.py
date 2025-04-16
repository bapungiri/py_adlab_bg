import matplotlib as mpl

stat_kw = dict(
    text_format="star",
    loc="inside",
    # verbose=True,
    fontsize=mpl.rcParams["axes.labelsize"],
    line_width=0.5,
    line_height=0.01,
    text_offset=0.2,
    # line_offset=0.2,
    # line_offset_to_group=0.9,
    pvalue_thresholds=[[0.05, "*"], [1, "ns"]],
    # pvalue_format={'star':[[0.05, "*"],[1, "ns"]]},
    # pvalue_format= {'correction_format': '{star} ({suffix})',
    #                           'fontsize': 'small',
    #                           'pvalue_format_string': '{:.3e}',
    #                           'show_test_name': True,
    #                         #   'simple_format_string': '{:.2f}',
    #                           'text_format': 'star',
    #                           'pvalue_thresholds': [
    #                               [1e-4, "*"],
    #                               [1e-3, "*"],
    #                               [1e-2, "*"],
    #                               [0.05, "*"],
    #                               [1, "ns"]]
    #                           },
    # color= 'r',
    # line_offset_to_box=0.2,
    # use_fixed_offset=True,
)
