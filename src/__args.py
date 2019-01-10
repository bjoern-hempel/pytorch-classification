def check_point_of_interest(fields, point_of_interest):
    if point_of_interest is not None:
        if not point_of_interest in fields:
            print('Unknown point of interest: {}'.format(point_of_interest))
            print('Allowed fields: "{}"'.format('", "'.join(fields)))
            exit()