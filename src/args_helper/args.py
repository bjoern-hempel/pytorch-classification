def check_point_of_interest(fields, args):
    """Check if the given field is within the point of interest list.
    Otherwise throw an error.
    """
    if args.point_of_interest is not None:
        if not args.point_of_interest in fields:
            print('Unknown point of interest: {}'.format(args.point_of_interest))
            print('Allowed fields: "{}"'.format('", "'.join(fields)))
            exit()


def prepare_args(args, output_modes):
    args.fields = args.fields.split(',')

    if args.output_mode is not None:
        args.fields = output_modes[args.output_mode]['fields']

    return args


def get_filters(args):
    filters = []
    for filter_arr in args.filter:
        if len(filter_arr) == 1:
            filter_arr = filter_arr[0].split('=')

        if len(filter_arr) == 2:
            filters.append({filter_arr[0]: filter_arr[1]})
            continue

    return filters
